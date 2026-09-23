import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from alphaproof.core.config import Config
from alphaproof.core.game import Game, extract_transitions
from alphaproof.training.sft import load_examples


Transition = tuple[str, str, float]
TokenizedTransition = tuple[torch.Tensor, torch.Tensor, float]


class ReplayBuffer:
    """Persistent replay storage for state-action-value training triples."""

    def __init__(self, config: Config, path: Path):
        """Initialize replay limits and restore existing transitions."""
        self.window_size = config.window_size
        self.batch_size = (
            config.batch_size * config.gradient_accumulation_steps
        )
        sft_batch_size = self.batch_size * config.sft_fraction
        if not math.isclose(sft_batch_size, round(sft_batch_size)):
            raise ValueError(
                'effective batch size * sft_fraction must be a whole number.'
            )
        self.sft_batch_size = round(sft_batch_size)
        if not 0 < self.sft_batch_size < self.batch_size:
            raise ValueError('Each batch must contain both SFT and replay data.')
        self.replay_batch_size = self.batch_size - self.sft_batch_size
        self.max_state_length = config.max_state_length
        self.max_action_length = config.max_action_length
        if not 0 < config.validation_fraction < 1:
            raise ValueError('validation_fraction must be between zero and one.')
        self.validation_stride = round(1 / config.validation_fraction)
        self.validation_window_size = max(
            1, round(self.window_size * config.validation_fraction)
        )
        self.path = path
        self.path.touch(exist_ok=True)
        self.buffer: list[Transition] = []
        self.validation_buffer: list[Transition] = []
        self.transition_ids: set[str] = set()
        self.transition_count = 0
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        sft_examples, sft_stats = load_examples(
            config.sft_dataset_path,
            self.tokenizer,
            self.max_state_length,
            self.max_action_length,
            None,
        )
        self.sft_buffer = [
            (example.state, example.action, example.value_target)
            for example in sft_examples
        ]
        if not self.sft_buffer:
            raise ValueError('The SFT dataset contains no usable transitions.')
        self._load()

    def save_game(self, game: Game) -> None:
        """Add solved-game transitions to the replay window and JSONL file."""
        for observation, action, value in extract_transitions(game.root):
            transition_id = f'actor:{self.transition_count + 1}'
            self._save_transition(
                transition_id,
                (str(observation), str(action), value),
                None,
            )

    def add_transitions(self, records: list[dict[str, Any]]) -> int:
        """Persist new externally generated transitions."""
        added = 0
        for record in records:
            transition_id = str(record['transition_id'])
            if transition_id in self.transition_ids:
                continue
            transition = (
                str(record['state']),
                str(record['action']),
                float(record['value']),
            )
            metadata = {
                name: record[name]
                for name in (
                    'batch_id',
                    'request_id',
                    'theorem_id',
                    'source',
                    'attempt',
                    'index',
                )
            }
            self._save_transition(transition_id, transition, metadata)
            added += 1
        return added

    def __len__(self) -> int:
        """Return the number of train transitions in the replay window."""
        return len(self.buffer)

    def sample_batch(self) -> list[TokenizedTransition]:
        """Sample a batch containing the configured SFT and replay mix."""
        batch = [
            self._tokenize_transition(random.choice(self.sft_buffer))
            for _ in range(self.sft_batch_size)
        ]
        batch.extend(
            self._tokenize_transition(transition)
            for transition in random.sample(
                self.buffer, self.replay_batch_size
            )
        )
        random.shuffle(batch)
        return batch

    def sample_transition(self) -> TokenizedTransition:
        """Sample and tokenize one training transition."""
        if not self.buffer:
            raise ValueError('Cannot sample from an empty replay buffer.')
        return self._tokenize_transition(random.choice(self.buffer))

    def validation_batch(self, batch_size: int) -> list[TokenizedTransition]:
        """Return a fixed held-out replay batch."""
        return [
            self._tokenize_transition(transition)
            for transition in self.validation_buffer[:batch_size]
        ]

    def _save_transition(
        self,
        transition_id: str,
        transition: Transition,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Assign and persist one transition."""
        self.transition_ids.add(transition_id)
        self.transition_count += 1
        validation = self.transition_count % self.validation_stride == 0
        target = self.validation_buffer if validation else self.buffer
        target.append(transition)
        if validation:
            self.validation_buffer = self.validation_buffer[
                -self.validation_window_size:
            ]
        else:
            self.buffer = self.buffer[-self.window_size:]

        state, action, value = transition
        record: dict[str, Any] = {
            'transition_id': transition_id,
            'state': state,
            'action': action,
            'value': value,
            'validation': validation,
        }
        if metadata is not None:
            record.update(metadata)
        with self.path.open('a', encoding='utf-8') as replay_file:
            replay_file.write(json.dumps(record) + '\n')

    def _load(self) -> None:
        """Restore the replay window from JSONL."""
        if not self.path.exists():
            return
        with self.path.open(encoding='utf-8') as replay_file:
            for line in replay_file:
                record = json.loads(line)
                self.transition_ids.add(str(record['transition_id']))
                transition = (
                    str(record['state']),
                    str(record['action']),
                    float(record['value']),
                )
                if record['validation']:
                    self.validation_buffer.append(transition)
                else:
                    self.buffer.append(transition)
                self.transition_count += 1
        self.buffer = self.buffer[-self.window_size:]
        self.validation_buffer = self.validation_buffer[
            -self.validation_window_size:
        ]

    def _tokenize_transition(
        self,
        transition: Transition,
    ) -> TokenizedTransition:
        state, action, value = transition
        return (
            self._tokenize(state, self.max_state_length),
            self._tokenize(action, self.max_action_length),
            value,
        )

    def _tokenize(self, text: str, max_length: int) -> torch.Tensor:
        """Tokenize text with fixed padding for replay batches."""
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return encoded.input_ids.squeeze(0).long()
