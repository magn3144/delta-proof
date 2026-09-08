import argparse
import math
import uuid
from pathlib import Path

from alphaproof.core.config import (
    Config,
    RL_PRECISIONS,
    load_experiment_config,
    rl_config_from_dict,
)
from alphaproof.core.paths import RUNS_DIR
from alphaproof.training.run_config import (
    changed_config_fields,
    has_run_config,
    load_run_config,
    save_run_config,
)
from alphaproof.training.run_logger import RunLogger, initialize_wandb
from alphaproof.training.train import alphaproof_train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse RL training arguments."""
    parser = argparse.ArgumentParser(description='Train AlphaProof with RL.')
    parser.add_argument('run_name', help='Directory name under data/runs.')
    parser.add_argument('config_path', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args(argv)
    if Path(args.run_name).name != args.run_name:
        parser.error('run_name must be a single directory name')
    if args.override and not args.resume:
        parser.error('--override requires --resume')
    if not args.config_path.is_file():
        parser.error(f'experiment YAML does not exist: {args.config_path}')
    return args


def validate_config(config: Config) -> None:
    """Validate relationships in a resolved RL configuration."""
    positive = (
        'num_simulations',
        'batch_size',
        'num_actors',
        'num_games_per_actor',
        'max_concurrent_lean_imports',
        'inference_num_gpus',
        'inference_batch_size',
        'num_sampled_actions',
        'rollout_max_action_length',
        'training_iterations',
        'checkpoint_interval',
        'theorem_validation_interval_games',
    )
    for name in positive:
        if getattr(config, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if config.training_steps < 0:
        raise ValueError('training_steps cannot be negative.')
    if config.lr <= 0:
        raise ValueError('lr must be positive.')
    if config.value_weight < 0:
        raise ValueError('value_weight cannot be negative.')
    if config.inference_batch_timeout < 0:
        raise ValueError('inference_batch_timeout cannot be negative.')
    if not 0 <= config.disprove_rate <= 1:
        raise ValueError('disprove_rate must be between zero and one.')
    if not 0 < config.sft_fraction < 1:
        raise ValueError('sft_fraction must be between zero and one.')
    if config.dtype not in RL_PRECISIONS:
        raise ValueError(f'dtype must be one of {RL_PRECISIONS}.')
    if config.wandb_mode not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb_mode must be online, offline, or disabled.')
    if not math.isclose(
        config.batch_size * config.sft_fraction,
        round(config.batch_size * config.sft_fraction),
    ):
        raise ValueError('batch_size * sft_fraction must be a whole number.')


def validate_config_paths(config: Config) -> None:
    """Validate files and directories required by RL training."""
    if config.sft_run_dir is None:
        raise ValueError('Set sft_run_dir in the RL configuration.')
    for dataset_path in (
        config.train_dataset_path,
        config.validation_dataset_path,
        config.test_dataset_path,
    ):
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f'Theorem dataset split does not exist: {dataset_path}'
            )
    if not config.sft_dataset_path.is_file():
        raise FileNotFoundError(
            f'SFT dataset does not exist: {config.sft_dataset_path}'
        )
    if not (config.sft_run_dir / 'model_source').is_dir():
        raise FileNotFoundError('SFT model_source directory does not exist.')
    if not (config.sft_run_dir / 'network_params.pt').is_file():
        raise FileNotFoundError('SFT network_params.pt does not exist.')


def prepare_run(args: argparse.Namespace) -> tuple[Config, Path, str]:
    """Create a new run or restore its saved configuration."""
    run_dir = RUNS_DIR / args.run_name

    if args.resume:
        saved = load_run_config(run_dir)
        saved_config = rl_config_from_dict(saved['config'], args.run_name)
        config = load_experiment_config(
            args.config_path,
            args.run_name,
            saved_config.seed,
        ).rl
        validate_config(config)
        validate_config_paths(config)
        changed_fields = changed_config_fields(saved_config, config)
        if changed_fields and not args.override:
            names = ', '.join(changed_fields)
            raise ValueError(
                f'Configuration differs for: {names}. Pass --override to '
                'resume with these values.'
            )
        if changed_fields:
            save_run_config(run_dir, config, saved['wandb_run_id'])
        return config, run_dir, saved['wandb_run_id']

    if has_run_config(run_dir):
        raise FileExistsError(f'Run already exists: {run_dir}')
    config = load_experiment_config(args.config_path, args.run_name).rl
    validate_config(config)
    validate_config_paths(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run_id = uuid.uuid4().hex
    save_run_config(run_dir, config, wandb_run_id)
    return config, run_dir, wandb_run_id


def main() -> None:
    """Run or resume AlphaProof reinforcement learning."""
    args = parse_args()
    config, run_dir, wandb_run_id = prepare_run(args)
    logger = RunLogger(
        run_dir,
        config.reward_window,
        initialize_wandb(
            args.run_name,
            run_dir,
            args.resume,
            wandb_run_id,
            config,
        ),
    )
    try:
        alphaproof_train(config, run_dir, args.resume, logger)
    except BaseException as error:
        logger.log_crash(error)
        logger.finish(exit_code=1)
        raise
    logger.finish(exit_code=0)


if __name__ == '__main__':
    main()
