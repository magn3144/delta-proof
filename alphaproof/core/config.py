import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast

import yaml

from alphaproof.core.environment import Environment
from alphaproof.core.paths import LEAN_PROJECT_DIR, PROJECT_ROOT
from leantree import LeanProject


RL_PRECISIONS = ('float32', 'bfloat16', 'mixed')
WandbMode = Literal['online', 'offline', 'disabled']
DEFAULT_EXPERIMENT_PATH = (
    PROJECT_ROOT / 'alphaproof' / 'yaml' / 'codet5p_770m_l40s.yaml'
)


@dataclass(frozen=True)
class SFTConfig:
    """Supervised fine-tuning settings loaded from an experiment file."""

    train_input: Path
    validation_input: Path
    model: Path
    epochs: int
    checkpoints_per_epoch: int
    num_pairs: int | None
    num_validation_pairs: int | None
    batch_size: int
    learning_rate: float
    value_weight: float
    max_state_length: int
    max_action_length: int
    rollout_max_action_length: int
    num_sampled_actions: int
    num_value_bins: int
    max_grad_norm: float
    log_every: int
    validation_interval: int
    validation_samples: int
    wandb_project: str
    wandb_name: str | None
    wandb_mode: WandbMode
    seed: int
    device: str
    dtype: str

    @property
    def tokenizer_model(self) -> str:
        return str(self.model)

    @property
    def lr(self) -> float:
        return self.learning_rate


@dataclass
class Config:
    """AlphaProof settings loaded from an experiment file."""

    num_simulations: int
    batch_size: int
    num_actors: int
    num_games_per_actor: int
    max_concurrent_lean_imports: int
    inference_num_gpus: int
    inference_batch_size: int
    inference_batch_timeout: float
    num_sampled_actions: int
    tactic_timeout: float
    final_check_timeout: float
    seed: int
    debug: bool
    lr: float
    dtype: str
    tokenizer_model: str
    dataset_dir: Path
    sft_dataset_path: Path
    sft_fraction: float
    disprove_rate: float
    sft_run_dir: Path | None
    max_state_length: int
    max_action_length: int
    rollout_max_action_length: int
    training_steps: int
    training_iterations: int
    checkpoint_interval: int
    window_size: int
    value_weight: float
    validation_fraction: float
    validation_batch_size: int
    validation_interval: int
    theorem_validation_interval_games: int
    theorem_validation_num_theorems: int
    log_interval: int
    reward_window: int
    wandb_project: str
    wandb_entity: str | None
    wandb_tags: tuple[str, ...]
    wandb_name: str | None
    wandb_mode: WandbMode
    pb_c_base: float
    pb_c_init: float
    value_discount: float
    prior_temperature: float
    c_and: float
    unvisited_value_penalty: float
    no_legal_actions_value: float
    ps_c: float
    ps_alpha: float
    num_value_bins: int
    mm_trust_count: int
    mm_fully_decided_trust_count: int
    mm_proved_weight: float
    mm_undecided_weight: float
    mm_simulation_failure_window: int
    mm_simulation_failure_multiplier: float
    mm_max_num_simulations: int
    run_id: int | str = field(init=False)
    environment_ctor: Callable[[], Environment] = field(init=False, repr=False)
    train_dataset_path: Path = field(init=False)
    validation_dataset_path: Path = field(init=False)
    test_dataset_path: Path = field(init=False)
    initial_params_path: Path | None = field(init=False)
    num_games: int = field(init=False)
    mm_disprove_rate: float = field(init=False)

    def __post_init__(self) -> None:
        self.run_id = 0
        self.environment_ctor = lambda: Environment(
            LeanProject(str(LEAN_PROJECT_DIR))
        )
        self.train_dataset_path = self.dataset_dir / 'train.jsonl'
        self.validation_dataset_path = self.dataset_dir / 'validation.jsonl'
        self.test_dataset_path = self.dataset_dir / 'test.jsonl'
        self.num_games = self.num_games_per_actor
        self.mm_disprove_rate = self.disprove_rate
        if self.sft_run_dir is None:
            self.initial_params_path = None
        else:
            self.tokenizer_model = str(self.sft_run_dir / 'model_source')
            self.initial_params_path = self.sft_run_dir / 'network_params.pt'


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete SFT and RL configuration for one experiment."""

    sft: SFTConfig
    rl: Config


def _resolve_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    if '$' in expanded:
        raise ValueError(f'Environment variable in {value!r} is not set')
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sft_config_from_dict(values: dict[str, Any]) -> SFTConfig:
    """Build a strict SFT configuration from serialized values."""
    data = dict(values)
    for name in ('train_input', 'validation_input', 'model'):
        data[name] = _resolve_path(data[name])
    return SFTConfig(**data)


def rl_config_from_dict(
    values: dict[str, Any],
    run_id: int | str = 0,
) -> Config:
    """Build a strict RL configuration from serialized values."""
    data = dict(values)
    data['dataset_dir'] = _resolve_path(data['dataset_dir'])
    data['sft_dataset_path'] = _resolve_path(data['sft_dataset_path'])
    data['tokenizer_model'] = str(_resolve_path(data['tokenizer_model']))
    if data['sft_run_dir'] is not None:
        data['sft_run_dir'] = _resolve_path(data['sft_run_dir'])
    data['wandb_tags'] = tuple(data['wandb_tags'])
    if data['seed'] is None:
        data['seed'] = secrets.randbits(63)
    config = Config(**data)
    config.run_id = run_id
    return config


def load_experiment_config(
    path: Path,
    run_id: int | str = 0,
    seed: int | None = None,
) -> ExperimentConfig:
    """Load both required sections of an experiment YAML file."""
    with path.open(encoding='utf-8') as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise TypeError('Experiment YAML must contain a mapping.')
    sections = cast(dict[str, Any], values['deltaproof'])
    if set(sections) != {'sft', 'rl'}:
        raise ValueError('The deltaproof section must contain exactly sft and rl sections.')
    if not isinstance(sections['sft'], dict) or not isinstance(
        sections['rl'], dict
    ):
        raise TypeError('The sft and rl sections must be mappings.')
    rl_values = dict(cast(dict[str, Any], sections['rl']))
    if rl_values['seed'] is None and seed is not None:
        rl_values['seed'] = seed
    return ExperimentConfig(
        sft=sft_config_from_dict(cast(dict[str, Any], sections['sft'])),
        rl=rl_config_from_dict(rl_values, run_id),
    )


def serializable_config(config: Config | SFTConfig) -> dict[str, Any]:
    """Convert source configuration fields to JSON-compatible values."""
    excluded = {
        'run_id',
        'environment_ctor',
        'train_dataset_path',
        'validation_dataset_path',
        'test_dataset_path',
        'initial_params_path',
        'num_games',
        'mm_disprove_rate',
    }
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
        if name not in excluded
    }
