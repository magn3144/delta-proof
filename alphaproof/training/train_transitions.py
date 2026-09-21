import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from alphaproof.core.config import Config, load_experiment_config, rl_config_from_dict
from alphaproof.core.network import Network
from alphaproof.training.randomness import seed_everything
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.rl_cli import validate_config, validate_config_paths
from alphaproof.training.run_config import (
    changed_config_fields,
    has_run_config,
    load_run_config,
    save_run_config,
)
from alphaproof.training.run_logger import (
    RunLogger,
    initialize_stp_wandb,
    load_stp_wandb_settings,
)
from alphaproof.training.shared_storage import SharedStorage
from alphaproof.training.train import REPLAY_FILE, train_network


BATCHES_FILE = 'transition_batches.json'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one externally generated learner batch."""
    parser = argparse.ArgumentParser(
        description='Train AlphaProof on externally generated transitions.'
    )
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--num-steps', type=int, required=True)
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f'Experiment YAML does not exist: {args.config}')
    if not args.input.is_file():
        parser.error(f'Transition input does not exist: {args.input}')
    if args.num_steps < 1:
        parser.error('--num-steps must be positive')
    return args


def prepare_run(
    config_path: Path,
    run_dir: Path,
    wandb_run_id: str,
) -> Config:
    """Create an external-transition run or restore its configuration."""
    resume = has_run_config(run_dir)
    if resume:
        saved = load_run_config(run_dir)
        saved_config = rl_config_from_dict(saved['config'], run_dir.name)
        config = load_experiment_config(
            config_path,
            run_dir.name,
            saved_config.seed,
        ).rl
        validate_config(config)
        validate_config_paths(config)
        changed_fields = changed_config_fields(saved_config, config)
        if changed_fields:
            names = ', '.join(changed_fields)
            raise ValueError(f'Configuration differs for: {names}.')
        if saved['wandb_run_id'] != wandb_run_id:
            raise ValueError('The saved W&B run ID differs from the STP run ID.')
    else:
        config = load_experiment_config(config_path, run_dir.name).rl
        validate_config(config)
        validate_config_paths(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_run_config(run_dir, config, wandb_run_id)
    return config


def sha256(path: Path) -> str:
    """Hash one immutable transition batch."""
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_batches(path: Path) -> dict[str, dict[str, Any]]:
    """Load learner-batch progress."""
    if not path.is_file():
        return {}
    with path.open(encoding='utf-8') as batches_file:
        return json.load(batches_file)


def save_batches(path: Path, batches: dict[str, dict[str, Any]]) -> None:
    """Atomically persist learner-batch progress."""
    temporary_path = path.with_suffix('.tmp')
    with temporary_path.open('w', encoding='utf-8') as batches_file:
        json.dump(batches, batches_file, indent=2)
        batches_file.write('\n')
    temporary_path.replace(path)


def load_transition_records(path: Path) -> list[dict[str, Any]]:
    """Load one transition JSONL batch."""
    with path.open(encoding='utf-8') as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def train_batch(
    args: argparse.Namespace,
    config: Config,
    run_dir: Path,
    logger: RunLogger,
) -> None:
    """Ingest and train one idempotent transition batch."""
    seed_everything(config.seed)
    storage = SharedStorage(run_dir)
    replay_buffer = ReplayBuffer(config, run_dir / REPLAY_FILE)
    network = Network(config)
    latest_path = storage.checkpoints_dir / 'latest.pt'
    if latest_path.is_file():
        step = storage.load_latest_checkpoint(network)
    else:
        if config.initial_params_path is None:
            raise ValueError('An SFT run is required for a new RL run.')
        network.load_params(config.initial_params_path)
        step = 0
        storage.save_checkpoint(step, network)
        storage.save_latest_checkpoint(step, network)

    batches_path = run_dir / BATCHES_FILE
    batches = load_batches(batches_path)
    input_sha256 = sha256(args.input)
    batch = batches.get(args.batch_id)
    if batch is None:
        batch = {
            'input': str(args.input.resolve()),
            'input_sha256': input_sha256,
            'num_steps': args.num_steps,
            'start_step': step,
            'target_step': step + args.num_steps,
            'ingested': False,
            'complete': False,
        }
        batches[args.batch_id] = batch
        save_batches(batches_path, batches)
    else:
        if batch['input_sha256'] != input_sha256:
            raise ValueError(f'Batch {args.batch_id} has different input data.')
        if batch['num_steps'] != args.num_steps:
            raise ValueError(f'Batch {args.batch_id} has a different step count.')

    if batch['complete']:
        print(f'Learner batch {args.batch_id} is already complete.', flush=True)
        return
    if step > batch['target_step']:
        raise ValueError('Latest checkpoint is ahead of this learner batch.')

    records = load_transition_records(args.input)
    transition_ids = [str(record['transition_id']) for record in records]
    if len(transition_ids) != len(set(transition_ids)):
        raise ValueError('Transition IDs must be unique within one input batch.')
    if any(str(record['batch_id']) != args.batch_id for record in records):
        raise ValueError('Transition batch_id does not match --batch-id.')
    added = replay_buffer.add_transitions(records)
    batch['ingested'] = True
    batch['transition_count'] = len(records)
    batch['new_transition_count'] = added
    save_batches(batches_path, batches)
    if len(replay_buffer) < replay_buffer.replay_batch_size:
        raise ValueError(
            f'Replay buffer has {len(replay_buffer)} training transitions; '
            f'{replay_buffer.replay_batch_size} are required.'
        )

    steps_to_run = int(batch['target_step']) - step
    if steps_to_run:
        step = train_network(
            config,
            network,
            storage,
            replay_buffer,
            step,
            steps_to_run,
            logger,
        )
    if step % config.checkpoint_interval != 0:
        storage.save_checkpoint(step, network)
    storage.save_latest_checkpoint(step, network)
    batch['complete'] = True
    save_batches(batches_path, batches)
    print(
        f'Completed learner batch {args.batch_id} at step {step} with '
        f'{added} new transitions.',
        flush=True,
    )


def main() -> None:
    """Train AlphaProof from one external transition file."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config_path = args.config.resolve()
    wandb_settings = load_stp_wandb_settings(config_path, run_dir)
    config = prepare_run(
        config_path,
        run_dir,
        wandb_settings['id'],
    )
    batches = load_batches(run_dir / BATCHES_FILE)
    batch = batches.get(args.batch_id)
    if batch is not None and batch['complete']:
        if batch['input_sha256'] != sha256(args.input):
            raise ValueError(f'Batch {args.batch_id} has different input data.')
        if batch['num_steps'] != args.num_steps:
            raise ValueError(f'Batch {args.batch_id} has a different step count.')
        print(f'Learner batch {args.batch_id} is already complete.', flush=True)
        return

    logger = RunLogger(
        run_dir,
        config.reward_window,
        initialize_stp_wandb(wandb_settings, config),
    )
    try:
        train_batch(args, config, run_dir, logger)
    except BaseException as error:
        logger.log_crash(error)
        logger.finish(exit_code=1)
        raise
    logger.finish(exit_code=0)


if __name__ == '__main__':
    main()
