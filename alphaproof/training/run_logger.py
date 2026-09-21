import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import wandb
import yaml

from alphaproof.core.config import Config, serializable_config
from alphaproof.core.game import Game
from alphaproof.inference.parallel import InferenceBatchStats
from alphaproof.training.resource_monitor import (
    ResourceMonitor,
    start_resource_monitor,
)
from alphaproof.training.run_diagnostics import RunDiagnostics


RESULTS_FILE = 'results.jsonl'
TIMINGS_FILE = 'timings.jsonl'
VALIDATION_RESULTS_FILE = 'validation_results.jsonl'


def load_stp_wandb_settings(config_path: Path, run_dir: Path) -> dict[str, Any]:
    """Load the shared STP W&B run settings from the experiment YAML."""
    with config_path.open(encoding='utf-8') as config_file:
        values = yaml.safe_load(config_file)
    tracker = values['training']['trainer']['tracker']
    experiment_dir = run_dir.parent
    run_id = (experiment_dir / 'conjecturer_wandb_id.txt').read_text(
        encoding='utf-8'
    ).strip()
    return {
        'entity': tracker['entity'],
        'project': tracker['project'],
        'name': tracker['name'] + '-conjecturer-metrics',
        'tags': tracker['tags'],
        'id': run_id,
        'dir': experiment_dir,
    }


def define_metrics(wandb_run: Any) -> None:
    """Configure metric axes shared by all AlphaProof W&B runs."""
    wandb_run.define_metric('actor/game')
    wandb_run.define_metric('actor/*', step_metric='actor/game')
    wandb_run.define_metric('learner/step')
    wandb_run.define_metric('train/*', step_metric='learner/step')
    wandb_run.define_metric('replay_validation/*', step_metric='learner/step')
    wandb_run.define_metric('validation/game')
    wandb_run.define_metric('validation/*', step_metric='validation/game')
    wandb_run.define_metric('inference/*')
    wandb_run.define_metric('resources/*')


def initialize_wandb(
    run_name: str,
    run_dir: Path,
    resume: bool,
    wandb_run_id: str,
    config: Config,
) -> Any:
    """Initialize W&B with the run's saved settings."""
    wandb_run: Any = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name or run_name,
        id=wandb_run_id,
        tags=config.wandb_tags,
        mode=config.wandb_mode,
        dir=run_dir,
        resume='allow' if resume else 'never',
        config=serializable_config(config),
        settings=wandb.Settings(
            finish_timeout=60.0,
            finish_timeout_raises=True,
        ),
    )
    define_metrics(wandb_run)
    return wandb_run


def initialize_stp_wandb(
    settings: dict[str, Any],
    config: Config,
) -> Any:
    """Resume the W&B run owned by the surrounding STP experiment."""
    wandb_run: Any = wandb.init(
        project=settings['project'],
        entity=settings['entity'],
        name=settings['name'],
        id=settings['id'],
        tags=settings['tags'],
        mode=config.wandb_mode,
        dir=settings['dir'],
        resume='must',
        config=serializable_config(config),
        settings=wandb.Settings(
            finish_timeout=60.0,
            finish_timeout_raises=True,
        ),
    )
    define_metrics(wandb_run)
    return wandb_run


class RunLogger:
    """Persist game results and isolate all Weights & Biases logging."""

    def __init__(
        self,
        run_dir: Path,
        reward_window: int,
        wandb_run: Any,
    ):
        self.results_path = run_dir / RESULTS_FILE
        self.results_path.touch(exist_ok=True)
        self.timings_path = run_dir / TIMINGS_FILE
        self.timings_path.touch(exist_ok=True)
        self.validation_results_path = run_dir / VALIDATION_RESULTS_FILE
        self.validation_results_path.touch(exist_ok=True)
        self.reward_window = reward_window
        self.wandb_run = wandb_run
        self.resource_monitor: ResourceMonitor | None = start_resource_monitor(
            wandb_run
        )
        self.diagnostics = RunDiagnostics(run_dir)
        successes, rewards, proved_theorems = self._load_results()
        self.games_completed = len(successes)
        self.actor_requests_started = self.games_completed
        self.proved_theorems = proved_theorems
        self.recent_successes = deque(
            successes[-reward_window:], maxlen=reward_window
        )
        self.recent_rewards = deque(rewards[-reward_window:], maxlen=reward_window)
        self.validation_games = self._load_validation_games()

    def next_actor_request_id(self) -> str:
        """Return a unique actor request identifier for this process."""
        self.actor_requests_started += 1
        return f'train-{self.actor_requests_started}'

    def log_game_start(self, request_id: str, game: Game) -> None:
        """Persist the game being started before Lean is launched."""
        self.diagnostics.game_started(request_id, game)
        print(f'Starting {request_id}.', flush=True)

    def log_learner_start(self, start_step: int, num_steps: int) -> None:
        """Persist the learner range about to run."""
        self.diagnostics.learner_started(start_step, start_step + num_steps)

    def log_game(self, game: Game, replay_size: int) -> None:
        """Persist and log one actor result."""
        success = int(game.root.is_optimal)
        reward = int(game.root.value_target) if success else None
        self.recent_successes.append(success)
        rolling_success_rate = (
            sum(self.recent_successes) / len(self.recent_successes)
        )
        if reward is not None:
            self.recent_rewards.append(reward)
        rolling_reward = (
            sum(self.recent_rewards) / len(self.recent_rewards)
            if self.recent_rewards
            else None
        )
        self.games_completed += 1
        if success and not game.disprove:
            self.proved_theorems.add(game.theorem)
        record = {
            'game': self.games_completed,
            'theorem': game.theorem,
            'disprove': game.disprove,
            'success': success,
            'final_proof': game.final_proof,
            'error': game.error,
            'episode_reward': reward,
            'rolling_success_rate': rolling_success_rate,
            'rolling_average_reward': rolling_reward,
            'num_simulations': game.num_simulations,
            'replay_size': replay_size,
        }
        with self.results_path.open('a', encoding='utf-8') as results_file:
            results_file.write(json.dumps(record) + '\n')
        timing_record = {
            'game': self.games_completed,
            'theorem': game.theorem,
            'disprove': game.disprove,
            'success': success,
            **game.timings.record(),
        }
        with self.timings_path.open('a', encoding='utf-8') as timings_file:
            timings_file.write(json.dumps(timing_record) + '\n')
        metrics = {
            'actor/game': self.games_completed,
            'actor/success': success,
            'actor/rolling_success_rate': rolling_success_rate,
            'actor/unique_theorems_proved': len(self.proved_theorems),
            'actor/num_simulations': game.num_simulations,
            'actor/game_seconds': game.timings.total_seconds,
            'actor/setup_seconds': game.timings.setup_seconds,
            'actor/tactic_generation_seconds': (
                game.timings.tactic_generation_seconds
            ),
            'actor/tactic_execution_seconds': (
                game.timings.tactic_execution_seconds
            ),
            'actor/internal_action_seconds': (
                game.timings.internal_action_seconds
            ),
            'replay/train_size': replay_size,
        }
        if game.timings.final_verification_seconds is not None:
            metrics['actor/final_verification_seconds'] = (
                game.timings.final_verification_seconds
            )
        if reward is not None:
            metrics['actor/episode_reward'] = reward
        if rolling_reward is not None:
            metrics['actor/rolling_average_reward'] = rolling_reward
        self.wandb_run.log(metrics)
        message = (
            f'Game {self.games_completed}: success {success}, '
            f'rolling success rate {rolling_success_rate:.3f}'
        )
        if reward is not None:
            message += f', reward {reward}'
        if game.error is not None:
            message += f', error: {game.error}\nTheorem:\n{game.theorem}'
        print(message, flush=True)

    def log_training(
        self,
        step: int,
        train_loss: float,
        learning_rate: float,
        validation_loss: float | None,
        replay_size: int,
    ) -> None:
        """Log learner metrics."""
        metrics = {
            'learner/step': step,
            'train/loss': train_loss,
            'train/learning_rate': learning_rate,
            'replay/train_size': replay_size,
        }
        if validation_loss is not None:
            metrics['replay_validation/loss'] = validation_loss
        self.wandb_run.log(metrics)
        message = f'Step {step}: train loss {train_loss:.4f}'
        if validation_loss is not None:
            message += f', replay validation loss {validation_loss:.4f}'
        print(message, flush=True)

    def log_inference(self, stats: InferenceBatchStats) -> None:
        """Log aggregate GPU inference batching measurements."""
        average_queue_wait = (
            stats.queue_wait_seconds / stats.request_count
            if stats.request_count
            else 0.0
        )
        self.wandb_run.log({
            'actor/game': self.games_completed,
            'actor/inference_batches': stats.batch_count,
            'actor/inference_average_batch_size': stats.average_batch_size,
            'actor/inference_average_queue_wait_seconds': average_queue_wait,
            'actor/inference_model_seconds': stats.model_seconds,
        })
        print(
            f'Inference: {stats.batch_count} batches, average size '
            f'{stats.average_batch_size:.2f}, model time '
            f'{stats.model_seconds:.1f}s.',
            flush=True,
        )

    def log_inference_batch(self, batch_size: int) -> None:
        """Log the actual size of a completed inference batch."""
        self.wandb_run.log({'inference/batch_size': batch_size})

    def log_validation(
        self,
        game: int,
        games: Sequence[Game | None],
    ) -> None:
        """Persist and log one fixed-theorem validation pass."""
        solved_games = [
            validation_game
            for validation_game in games
            if validation_game is not None
            and validation_game.root.is_optimal
        ]
        solve_rate = len(solved_games) / len(games)
        average_reward = (
            sum(
                validation_game.root.value_target
                for validation_game in solved_games
            )
            / len(solved_games)
            if solved_games
            else None
        )
        record = {
            'game': game,
            'num_theorems': len(games),
            'num_solved': len(solved_games),
            'solve_rate': solve_rate,
            'average_reward': average_reward,
        }
        with self.validation_results_path.open(
            'a', encoding='utf-8'
        ) as validation_file:
            validation_file.write(json.dumps(record) + '\n')
        self.validation_games.add(game)

        metrics = {
            'validation/game': game,
            'validation/solve_rate': solve_rate,
            'validation/average_reward': average_reward,
        }
        self.wandb_run.log(metrics)

        message = f'Validation after game {game}: solve rate {solve_rate:.3f}'
        if average_reward is not None:
            message += f', average reward {average_reward:.3f}'
        print(message, flush=True)

    def log_crash(self, error: BaseException) -> None:
        """Persist a fatal error before W&B shutdown."""
        self.diagnostics.crash(error)

    def finish(self, exit_code: int) -> None:
        """Finish W&B without hiding an existing training failure."""
        if self.resource_monitor is not None:
            self.resource_monitor.stop()
        try:
            self.wandb_run.finish(exit_code=exit_code)
        except Exception as error:
            self.diagnostics.wandb_finish_failed(error)
            if exit_code == 0:
                raise
        else:
            if exit_code == 0:
                self.diagnostics.complete()
        finally:
            self.diagnostics.close()

    def _load_results(self) -> tuple[list[int], list[int], set[str]]:
        """Load completed successes and solved rewards when resuming."""
        if not self.results_path.exists():
            return [], [], set()
        successes = []
        rewards = []
        proved_theorems = set()
        with self.results_path.open(encoding='utf-8') as results_file:
            for line in results_file:
                record = json.loads(line)
                successes.append(int(record['success']))
                if record['success'] and not record['disprove']:
                    proved_theorems.add(str(record['theorem']))
                if record['episode_reward'] is not None:
                    rewards.append(int(record['episode_reward']))
        return successes, rewards, proved_theorems

    def _load_validation_games(self) -> set[int]:
        """Load completed validation points when resuming."""
        games = set()
        with self.validation_results_path.open(encoding='utf-8') as results_file:
            for line in results_file:
                games.add(int(json.loads(line)['game']))
        return games
