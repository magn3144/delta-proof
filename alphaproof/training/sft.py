"""Supervised fine-tuning of AlphaProof's CodeT5 policy and value network."""

import argparse
import gc
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, RobertaTokenizer

from alphaproof.core.config import (
    SFTConfig,
    load_experiment_config,
    sft_config_from_dict,
)
from alphaproof.core.network import Network
from alphaproof.core.paths import RUNS_DIR
from alphaproof.training.run_config import (
    changed_config_fields,
    has_run_config,
    load_run_config,
    save_run_config,
)
from alphaproof.training.sft_logger import SFTLogger


TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


@dataclass(frozen=True)
class TrainingExample:
    """One LeanTree transition with an AlphaProof value target."""

    state: str
    action: str
    value_target: float


@dataclass(frozen=True)
class DatasetStats:
    """Counts collected while validating and filtering an input JSONL."""

    records_read: int
    examples_kept: int
    states_too_long: int
    actions_too_long: int


class TransitionDataset(Dataset[TrainingExample]):
    """In-memory LeanTree transitions used for shuffled SFT batches."""

    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


class TransitionCollator:
    """Tokenize already length-checked states and tactics with dynamic padding."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(
        self, batch: list[TrainingExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = [example.state for example in batch]
        actions = [example.action for example in batch]
        observations = self.tokenizer(
            states,
            padding=True,
            truncation=False,
            return_tensors='pt',
        ).input_ids
        action_tokens = self.tokenizer(
            text_target=actions,
            padding=True,
            truncation=False,
            return_tensors='pt',
        ).input_ids
        value_targets = torch.tensor(
            [example.value_target for example in batch],
            dtype=torch.float32,
        )
        return observations.long(), action_tokens.long(), value_targets


def token_length(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count tokens without allowing the tokenizer to truncate the text."""
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        verbose=False,
    )
    return len(encoded['input_ids'])


def load_examples(
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_state_length: int,
    max_action_length: int,
    record_limit: int | None,
) -> tuple[list[TrainingExample], DatasetStats]:
    """Validate records and reject examples that would require truncation."""
    examples = []
    records_read = 0
    states_too_long = 0
    actions_too_long = 0

    with path.open(encoding='utf-8') as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if record_limit is not None and records_read >= record_limit:
                break
            records_read += 1
            record = json.loads(line)
            state = record.get('state')
            action = record.get('action')
            proof_depth = record.get('proof_depth')
            if not isinstance(state, str) or not state.strip():
                raise ValueError(
                    f'Expected a non-empty string state on line {line_number} of {path}.'
                )
            if not isinstance(action, str) or not action.strip():
                raise ValueError(
                    f'Expected a non-empty string action on line {line_number} of {path}.'
                )
            if (
                not isinstance(proof_depth, int)
                or isinstance(proof_depth, bool)
                or proof_depth < 1
            ):
                raise ValueError(
                    f'Expected a positive integer proof_depth on line '
                    f'{line_number} of {path}.'
                )

            state_length = token_length(tokenizer, state)
            action_length = token_length(tokenizer, action)
            if state_length > max_state_length:
                states_too_long += 1
                continue
            if action_length > max_action_length:
                actions_too_long += 1
                continue
            examples.append(
                TrainingExample(
                    state=state.strip(),
                    action=action.strip(),
                    value_target=-float(proof_depth),
                )
            )

    return examples, DatasetStats(
        records_read=records_read,
        examples_kept=len(examples),
        states_too_long=states_too_long,
        actions_too_long=actions_too_long,
    )


def resolve_device(name: str) -> torch.device:
    """Resolve auto to CUDA, then Apple Silicon MPS, then CPU."""
    if name != 'auto':
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is not available.')
    return device


def make_network(config: SFTConfig, device: torch.device) -> Network:
    """Construct the existing AlphaProof network with SFT hyperparameters."""
    tokenizer = RobertaTokenizer(
        vocab=str(config.model / 'vocab.json'),
        merges=str(config.model / 'merges.txt'),
        model_max_length=config.max_state_length,
    )
    with patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer):
        network = Network(config)
    network.device = device
    network.to(device=device, dtype=TORCH_DTYPES[config.dtype])
    return network


def batch_losses(
    network: Network,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """Compute joint policy and proof-depth value losses."""
    observations, actions, value_targets = batch
    observations = observations.to(network.device)
    actions = actions.to(network.device)
    value_targets = value_targets.to(network.device)
    output = network(observations, actions)
    policy_loss = output.policy_loss.float()
    value_loss = network.value_loss(
        output.value_logits.float(),
        value_targets.float(),
    )
    total_loss = policy_loss + value_weight * value_loss
    return total_loss, policy_loss, value_loss, output


def clear_oom_memory(network: Network) -> None:
    """Release gradients and unused CUDA memory after an OOM."""
    network.optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if network.device.type == 'cuda':
        torch.cuda.empty_cache()


def train_batch(
    network: Network,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    """Apply one optimizer update and return detached component losses."""
    network.optimizer.zero_grad(set_to_none=True)
    loss, policy_loss, value_loss, _ = batch_losses(
        network,
        batch,
        args.value_weight,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
    network.optimizer.step()
    return (
        loss.detach().item(),
        policy_loss.detach().item(),
        value_loss.detach().item(),
    )


def train_epoch(
    network: Network,
    data_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    train_examples_per_epoch: int,
    validation_samples: int,
    run_dir: Path,
    args: argparse.Namespace,
    epoch: int,
    logger: SFTLogger,
) -> dict[str, float]:
    """Train for one epoch, logging periodic train and validation metrics."""
    network.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    examples_seen = 0
    oom_batches = 0

    for step, batch in enumerate(data_loader, start=1):
        batch_size = batch[0].shape[0]
        batch_metrics = None
        oom_message = ''
        try:
            batch_metrics = train_batch(
                network,
                batch,
                args,
            )
        except torch.OutOfMemoryError as error:
            oom_message = str(error)

        if batch_metrics is None:
            oom_batches += 1
            clear_oom_memory(network)
            print(
                f'WARNING: OOM in training epoch {epoch}, step {step}/'
                f'{len(data_loader)}; skipped {batch_size} examples and cleared '
                f'the CUDA cache. {oom_message}',
                flush=True,
            )
            continue

        batch_loss, batch_policy_loss, batch_value_loss = batch_metrics
        examples_seen += batch_size
        total_loss += batch_loss * batch_size
        total_policy_loss += batch_policy_loss * batch_size
        total_value_loss += batch_value_loss * batch_size
        global_step = (epoch - 1) * len(data_loader) + step
        if step % args.log_every == 0:
            logger.log_training(
                global_step,
                batch_loss,
                batch_policy_loss,
                batch_value_loss,
                network.optimizer.param_groups[0]['lr'],
                (epoch - 1) * train_examples_per_epoch + examples_seen,
            )
            print(
                f'Epoch {epoch}/{args.epochs}, step {step}/{len(data_loader)}, '
                f'loss {total_loss / examples_seen:.4f}',
                flush=True,
            )
        if (
            global_step % args.validation_interval == 0
            and step < len(data_loader)
        ):
            validation_metrics = validate(
                network,
                validation_loader,
                args.value_weight,
            )
            logger.log_validation(
                global_step,
                validation_metrics,
                validation_samples,
            )
            print(
                f'Step {global_step}: validation loss '
                f"{validation_metrics['loss']:.4f}",
                flush=True,
            )
            network.train()
        for checkpoint_index in range(1, args.checkpoints_per_epoch):
            checkpoint_step = (
                checkpoint_index * len(data_loader) + args.checkpoints_per_epoch - 1
            ) // args.checkpoints_per_epoch
            if step == checkpoint_step:
                checkpoint_epoch = (
                    epoch - 1 + checkpoint_index / args.checkpoints_per_epoch
                )
                checkpoint_path = save_checkpoint(
                    run_dir,
                    network,
                    checkpoint_epoch,
                )
                print(f'Saved {checkpoint_path}', flush=True)

    if oom_batches:
        print(
            f'Epoch {epoch}: skipped {oom_batches} training batches due to OOM.',
            flush=True,
        )
    return {
        'loss': total_loss / examples_seen if examples_seen else float('nan'),
        'policy_loss': (
            total_policy_loss / examples_seen if examples_seen else float('nan')
        ),
        'value_loss': (
            total_value_loss / examples_seen if examples_seen else float('nan')
        ),
        'oom_batches': float(oom_batches),
    }


def validation_batch_metrics(
    network: Network,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    value_weight: float,
    pad_token_id: int | None,
) -> tuple[float, float, float, float, int]:
    """Evaluate one validation batch and return detached metric totals."""
    loss, policy_loss, value_loss, output = batch_losses(
        network,
        batch,
        value_weight,
    )
    actions = batch[1].to(network.device)
    value_targets = batch[2].to(network.device)
    predictions = output.policy_logits.argmax(dim=-1)
    if pad_token_id is None:
        correct = predictions.eq(actions).all(dim=-1)
    else:
        token_correct = predictions.eq(actions) | actions.eq(pad_token_id)
        correct = token_correct.all(dim=-1)

    value_probabilities = torch.softmax(
        output.value_logits.float(),
        dim=-1,
    )
    predicted_values = (
        value_probabilities * network.value_bins.float()
    ).sum(dim=-1)
    value_error = torch.abs(predicted_values - value_targets).sum().item()
    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        value_error,
        int(correct.sum().item()),
    )


def validate(
    network: Network,
    data_loader: DataLoader[Any],
    value_weight: float,
) -> dict[str, float]:
    """Evaluate losses, teacher-forced tactic accuracy, and value error."""
    network.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_value_error = 0.0
    exact_tactics = 0
    examples_seen = 0
    oom_batches = 0
    pad_token_id = network.model.config.pad_token_id

    with torch.no_grad():
        for step, batch in enumerate(data_loader, start=1):
            batch_size = batch[0].shape[0]
            batch_metrics = None
            oom_message = ''
            try:
                batch_metrics = validation_batch_metrics(
                    network,
                    batch,
                    value_weight,
                    pad_token_id,
                )
            except torch.OutOfMemoryError as error:
                oom_message = str(error)

            if batch_metrics is None:
                oom_batches += 1
                clear_oom_memory(network)
                print(
                    f'WARNING: OOM in validation step {step}/{len(data_loader)}; '
                    f'skipped {batch_size} examples and cleared the CUDA cache. '
                    f'{oom_message}',
                    flush=True,
                )
                continue

            (
                batch_loss,
                batch_policy_loss,
                batch_value_loss,
                batch_value_error,
                batch_exact_tactics,
            ) = batch_metrics
            examples_seen += batch_size
            exact_tactics += batch_exact_tactics
            total_value_error += batch_value_error
            total_loss += batch_loss * batch_size
            total_policy_loss += batch_policy_loss * batch_size
            total_value_loss += batch_value_loss * batch_size

    if oom_batches:
        print(
            f'Validation: skipped {oom_batches} batches due to OOM.',
            flush=True,
        )
    return {
        'loss': total_loss / examples_seen if examples_seen else float('nan'),
        'policy_loss': (
            total_policy_loss / examples_seen if examples_seen else float('nan')
        ),
        'value_loss': (
            total_value_loss / examples_seen if examples_seen else float('nan')
        ),
        'teacher_forced_tactic_accuracy': (
            exact_tactics / examples_seen if examples_seen else float('nan')
        ),
        'value_mae': (
            total_value_error / examples_seen if examples_seen else float('nan')
        ),
        'oom_batches': float(oom_batches),
    }


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Append one epoch of training and validation metrics."""
    with path.open('a', encoding='utf-8') as metrics_file:
        metrics_file.write(json.dumps(metrics) + '\n')


def save_checkpoint(
    run_dir: Path,
    network: Network,
    epoch: float,
) -> Path:
    """Save resumable training state and AlphaProof-compatible parameters."""
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch:g}.pt'
    torch.save(
        {
            'epoch': epoch,
            'network_params': network.params,
            'optimizer_state_dict': network.optimizer.state_dict(),
        },
        checkpoint_path,
    )
    parameters_path = run_dir / 'network_params.pt'
    temporary_path = run_dir / 'network_params.tmp'
    torch.save(network.params, temporary_path)
    temporary_path.replace(parameters_path)
    return checkpoint_path


def save_network_source(
    run_dir: Path,
    network: Network,
    base_model_dir: Path,
) -> Path:
    """Save lightweight model metadata for constructing the trained Network."""
    model_source_dir = run_dir / 'model_source'
    model_source_dir.mkdir(exist_ok=True)
    network.tokenizer.save_pretrained(model_source_dir)

    config = json.loads(
        (base_model_dir / 'config.json').read_text(encoding='utf-8')
    )
    config.pop('torch_dtype', None)
    config['dtype'] = str(next(network.parameters()).dtype).removeprefix('torch.')
    (model_source_dir / 'config.json').write_text(
        json.dumps(config, indent=2) + '\n',
        encoding='utf-8',
    )

    source_weights = (base_model_dir / 'pytorch_model.bin').resolve()
    linked_weights = model_source_dir / 'pytorch_model.bin'
    if linked_weights.exists() or linked_weights.is_symlink():
        linked_weights.unlink()
    linked_weights.symlink_to(source_weights)
    return model_source_dir


def load_latest_checkpoint(run_dir: Path, network: Network) -> float:
    """Restore the latest complete epoch, or return zero if none exists."""
    checkpoints = sorted((run_dir / 'checkpoints').glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        print(
            f'No checkpoints found under {run_dir}; restarting from epoch 1.',
            flush=True,
        )
        return 0
    checkpoint = torch.load(
        checkpoints[-1],
        map_location=network.device,
        weights_only=False,
    )
    network.params = checkpoint['network_params']
    network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return float(checkpoint['epoch'])


def print_dataset_stats(name: str, stats: DatasetStats) -> None:
    """Report exactly how many overlength records were rejected."""
    print(
        f'{name}: read {stats.records_read:,}, kept {stats.examples_kept:,}, '
        f'skipped {stats.states_too_long:,} long states and '
        f'{stats.actions_too_long:,} long actions',
        flush=True,
    )


def train(args: argparse.Namespace, config: SFTConfig) -> Path:
    """Run joint supervised policy and value training."""
    run_dir = RUNS_DIR / args.run_name
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    network = make_network(config, device)
    train_examples, train_stats = load_examples(
        args.train_input,
        network.tokenizer,
        args.max_state_length,
        args.max_action_length,
        args.num_pairs,
    )
    validation_examples, validation_stats = load_examples(
        args.validation_input,
        network.tokenizer,
        args.max_state_length,
        args.max_action_length,
        args.num_validation_pairs,
    )
    print_dataset_stats('Train', train_stats)
    print_dataset_stats('Validation', validation_stats)
    if not train_examples:
        raise ValueError('No training examples remain after length filtering.')
    if not validation_examples:
        raise ValueError('No validation examples remain after length filtering.')

    collator = TransitionCollator(network.tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TransitionDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        pin_memory=device.type == 'cuda',
    )
    validation_loader = DataLoader(
        TransitionDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == 'cuda',
    )
    frequent_validation_examples = random.Random(args.seed).sample(
        validation_examples,
        k=min(args.validation_samples, len(validation_examples)),
    )
    frequent_validation_loader = DataLoader(
        TransitionDataset(frequent_validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == 'cuda',
    )

    first_epoch = 1
    if args.resume:
        first_epoch = int(load_latest_checkpoint(run_dir, network)) + 1
    if first_epoch > args.epochs:
        model_source_dir = save_network_source(run_dir, network, args.model)
        print(
            f'Run already completed epoch {first_epoch - 1}; '
            f'model metadata is available at {model_source_dir}',
            flush=True,
        )
        return run_dir

    metrics_path = run_dir / 'metrics.jsonl'
    print(f'Training on {device}', flush=True)
    logger = SFTLogger(
        args.run_name,
        run_dir,
        args.resume,
        args.wandb_run_id,
        config,
    )
    try:
        for epoch in range(first_epoch, args.epochs + 1):
            training_metrics = train_epoch(
                network,
                train_loader,
                frequent_validation_loader,
                len(train_examples),
                len(frequent_validation_examples),
                run_dir,
                args,
                epoch,
                logger,
            )
            validation_metrics = validate(
                network, validation_loader, args.value_weight
            )
            global_step = epoch * len(train_loader)
            metrics = {
                'epoch': epoch,
                'train': training_metrics,
                'validation': validation_metrics,
            }
            append_metrics(metrics_path, metrics)
            logger.log_epoch(
                global_step,
                epoch,
                training_metrics,
                validation_metrics,
                len(validation_examples),
            )
            checkpoint_path = save_checkpoint(run_dir, network, epoch)
            print(
                f"Finished epoch {epoch}/{args.epochs}: train loss "
                f"{training_metrics['loss']:.4f}, validation loss "
                f"{validation_metrics['loss']:.4f}; saved {checkpoint_path}",
                flush=True,
            )
        model_source_dir = save_network_source(run_dir, network, args.model)
    except BaseException:
        logger.finish(exit_code=1)
        raise
    logger.finish(exit_code=0)

    print(
        f'Saved AlphaProof model metadata to {model_source_dir}',
        flush=True,
    )
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse supervised fine-tuning command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune AlphaProof on LeanTree transitions.'
    )
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


def validate_config(config: SFTConfig) -> None:
    """Validate a resolved SFT configuration."""
    positive = (
        'epochs',
        'checkpoints_per_epoch',
        'batch_size',
        'max_state_length',
        'max_action_length',
        'num_value_bins',
        'log_every',
        'validation_interval',
        'validation_samples',
    )
    for name in positive:
        if getattr(config, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if config.num_pairs is not None and config.num_pairs < 1:
        raise ValueError('num_pairs must be positive when set.')
    if config.num_validation_pairs is not None and config.num_validation_pairs < 1:
        raise ValueError('num_validation_pairs must be positive when set.')
    if config.learning_rate <= 0:
        raise ValueError('learning_rate must be positive.')
    if config.value_weight < 0:
        raise ValueError('value_weight cannot be negative.')
    if config.max_grad_norm <= 0:
        raise ValueError('max_grad_norm must be positive.')
    if config.device not in ('auto', 'cpu', 'cuda', 'mps'):
        raise ValueError('device must be auto, cpu, cuda, or mps.')
    if config.dtype not in TORCH_DTYPES:
        raise ValueError(f'dtype must be one of {tuple(TORCH_DTYPES)}.')
    if config.wandb_mode not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb_mode must be online, offline, or disabled.')
    if not config.train_input.is_file():
        raise FileNotFoundError(f'Training JSONL does not exist: {config.train_input}')
    if not config.validation_input.is_file():
        raise FileNotFoundError(
            f'Validation JSONL does not exist: {config.validation_input}'
        )
    if not config.model.is_dir():
        raise FileNotFoundError(f'Model directory does not exist: {config.model}')


def prepare_run(
    cli_args: argparse.Namespace,
) -> tuple[argparse.Namespace, SFTConfig]:
    """Create a new SFT run or restore its saved configuration."""
    run_dir = RUNS_DIR / cli_args.run_name
    if cli_args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f'SFT run does not exist: {run_dir}')
        saved = load_run_config(run_dir)
        saved_config = sft_config_from_dict(saved['config'])
        config = load_experiment_config(cli_args.config_path).sft
        wandb_run_id = saved['wandb_run_id']
        validate_config(config)
        changed_fields = changed_config_fields(saved_config, config)
        if changed_fields and not cli_args.override:
            names = ', '.join(changed_fields)
            raise ValueError(
                f'Configuration differs for: {names}. Pass --override to '
                'resume with these values.'
            )
        if changed_fields:
            save_run_config(run_dir, config, wandb_run_id)
    else:
        if has_run_config(run_dir):
            raise FileExistsError(f'SFT run already exists: {run_dir}')
        config = load_experiment_config(cli_args.config_path).sft
        wandb_run_id = uuid.uuid4().hex
        validate_config(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_run_config(run_dir, config, wandb_run_id)
    args = argparse.Namespace(
        **vars(config),
        run_name=cli_args.run_name,
        resume=cli_args.resume,
        wandb_run_id=wandb_run_id,
    )
    return args, config


def main() -> None:
    """Run supervised fine-tuning."""
    args, config = prepare_run(parse_args())
    run_dir = train(args, config)
    print(f'Training complete. Outputs saved under {run_dir}', flush=True)


if __name__ == '__main__':
    main()
