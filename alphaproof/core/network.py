import copy
import typing
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import parallel_apply
from transformers import AutoTokenizer, T5ForConditionalGeneration

from alphaproof.core.config import Config, SFTConfig
from alphaproof.core.environment import Action


Params = dict[str, torch.Tensor]
TORCH_DTYPES = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'mixed': torch.float32,
}


class NetworkTrainingOutput(typing.NamedTuple):
    """Output of the network during training."""
    value_logits: torch.Tensor
    policy_logits: torch.Tensor
    policy_loss: torch.Tensor


class NetworkSamplingOutput(typing.NamedTuple):
    """Output of the network when sampling actions."""
    action_logprobs: Dict[Action, float]
    value: float


class _InferenceModel(nn.Module):
    """Model replica used for one inference batch shard."""

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        value_head: nn.Linear,
        rollout_max_action_length: int,
        num_sampled_actions: int,
        mixed_precision: bool,
    ):
        super().__init__()
        self.model = model
        self.value_head = value_head
        self.rollout_max_action_length = rollout_max_action_length
        self.num_sampled_actions = num_sampled_actions
        self.mixed_precision = mixed_precision

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate tactics and value logits for one device-local shard."""
        with torch.no_grad(), torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=self.mixed_precision,
        ):
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_state = encoder_outputs.last_hidden_state
            mask = attention_mask.to(dtype=hidden_state.dtype).unsqueeze(-1)
            pooled_state = (
                (hidden_state * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1)
            )
            value_logits = self.value_head(pooled_state)

            generation_model = typing.cast(typing.Any, self.model)
            generated = generation_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_new_tokens=self.rollout_max_action_length,
                num_return_sequences=self.num_sampled_actions,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
            generated = typing.cast(typing.Any, generated)
            if generated.scores is None:
                raise ValueError('Expected generation output to include scores.')
            transition_scores = generation_model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                normalize_logits=True,
            )
            generated_tokens = generated.sequences[
                :, -transition_scores.shape[-1] :
            ]
            pad_token_id = typing.cast(int, self.model.config.pad_token_id)
            score_mask = generated_tokens.ne(pad_token_id)
            logprobs = transition_scores.masked_fill(
                ~score_mask,
                0.0,
            ).sum(dim=-1)

        return value_logits, generated.sequences, logprobs


class Network(nn.Module):
    """CodeT5+ policy and value network used by the training loop."""

    def __init__(self, config: Config | SFTConfig):
        """Initialize the model, value head, and optimizer."""
        super().__init__()

        inference_num_gpus = (
            config.inference_num_gpus if isinstance(config, Config) else 1
        )
        if inference_num_gpus < 1:
            raise ValueError('Inference GPU count must be positive.')
        available_gpus = torch.cuda.device_count()
        if isinstance(config, Config) and available_gpus < inference_num_gpus:
            raise RuntimeError(
                f'Requested {inference_num_gpus} inference GPUs, but only '
                f'{available_gpus} are available.'
            )

        self.num_value_bins = config.num_value_bins
        self.value_weight = config.value_weight
        self.max_state_length = config.max_state_length
        self.max_action_length = config.max_action_length
        self.mixed_precision = config.dtype == 'mixed'
        self.device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.tokenizer_model
        )
        self.value_head = nn.Linear(
            self.model.config.d_model,
            self.num_value_bins,
            dtype=self.model.dtype,
        )

        self.value_bins: torch.Tensor
        value_bins = torch.linspace(
            -float(self.num_value_bins - 1),
            0.0,
            steps=self.num_value_bins,
        )
        self.register_buffer('value_bins', value_bins)
        self.to(device=self.device, dtype=TORCH_DTYPES[config.dtype])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

        self._inference_models = []
        if isinstance(config, Config):
            self.rollout_max_action_length = config.rollout_max_action_length
            self.num_sampled_actions = config.num_sampled_actions
            primary_inference_model = _InferenceModel(
                self.model,
                self.value_head,
                self.rollout_max_action_length,
                self.num_sampled_actions,
                self.mixed_precision,
            )
            self._inference_models = [primary_inference_model]
            for gpu_index in range(1, inference_num_gpus):
                self._inference_models.append(
                    copy.deepcopy(primary_inference_model).to(f'cuda:{gpu_index}')
                )
        self._inference_replicas_stale = False

    @property
    def params(self) -> Params:
        """Return a PyTorch checkpoint compatible with shared storage."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }

    @params.setter
    def params(self, params: Params):
        """Load a PyTorch checkpoint from shared storage."""
        self.load_state_dict(params)
        self._inference_replicas_stale = True

    def load_params(self, path: Path) -> None:
        """Load network parameters saved by supervised fine-tuning."""
        params = torch.load(path, map_location='cpu', weights_only=True)
        self.params = typing.cast(Params, params)

    def _loss_fn(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Compute policy and value loss over a replay batch."""
        if not batch:
            raise ValueError('Cannot compute network loss for an empty batch.')

        observations = torch.stack(
            [observation for observation, _, _ in batch]
        ).to(self.device)
        actions = torch.stack([action for _, action, _ in batch]).to(
            self.device
        )
        value_targets = torch.tensor(
            [value_target for _, _, value_target in batch],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.mixed_precision,
        ):
            network_output = self.forward(observations, actions)
        return (
            network_output.policy_loss.float()
            + self.value_weight
            * self.value_loss(network_output.value_logits.float(), value_targets)
        )

    def value_loss(
        self, value_logits: torch.Tensor, value_targets: float | torch.Tensor
    ) -> torch.Tensor:
        """Calculate categorical value loss with linear interpolation bins."""
        value_bins = self.value_bins.to(
            device=value_logits.device,
            dtype=value_logits.dtype,
        )
        targets = torch.as_tensor(
            value_targets,
            dtype=value_logits.dtype,
            device=value_logits.device,
        )
        targets = targets.clamp(value_bins[0], value_bins[-1])
        lower_index = torch.searchsorted(value_bins, targets, right=True) - 1
        lower_index = lower_index.clamp(0, self.num_value_bins - 1)
        upper_index = (lower_index + 1).clamp(0, self.num_value_bins - 1)

        lower_bin = value_bins[lower_index]
        upper_bin = value_bins[upper_index]
        upper_weight = torch.where(
            upper_bin == lower_bin,
            torch.zeros_like(targets),
            (targets - lower_bin) / (upper_bin - lower_bin),
        )
        lower_weight = 1.0 - upper_weight

        target_distribution = torch.zeros_like(value_logits)
        target_distribution.scatter_add_(
            -1,
            lower_index.reshape(target_distribution.shape[:-1] + (1,)),
            lower_weight.reshape(target_distribution.shape[:-1] + (1,)),
        )
        target_distribution.scatter_add_(
            -1,
            upper_index.reshape(target_distribution.shape[:-1] + (1,)),
            upper_weight.reshape(target_distribution.shape[:-1] + (1,)),
        )

        log_probs = F.log_softmax(value_logits, dim=-1)
        return -(target_distribution * log_probs).sum(dim=-1).mean()

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> NetworkTrainingOutput:
        """Run the network for supervised tactic and value training."""
        observation = self._ensure_batch(observation).to(self.device)
        action = self._ensure_batch(action).to(self.device)
        labels = action.clone()

        pad_token_id = self.model.config.pad_token_id
        attention_mask = None
        if pad_token_id is not None:
            attention_mask = observation.ne(pad_token_id).long()
            labels = labels.masked_fill(labels == pad_token_id, -100)

        outputs = self.model(
            input_ids=observation,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        if outputs.loss is None:
            raise ValueError('Expected model output to include policy loss.')
        if outputs.encoder_last_hidden_state is None:
            raise ValueError(
                'Expected model output to include encoder hidden states.'
            )

        pooled_state = self._mean_pool_encoder_state(
            outputs.encoder_last_hidden_state,
            attention_mask,
        )
        value_logits = self.value_head(pooled_state)
        return NetworkTrainingOutput(
            value_logits=value_logits,
            policy_logits=outputs.logits,
            policy_loss=outputs.loss,
        )

    def sample(self, observation: str) -> NetworkSamplingOutput:
        """Return sampled tactics and a value estimate for search."""
        return self.sample_batch([observation])[0]

    def sample_batch(
        self,
        observations: list[str],
    ) -> list[NetworkSamplingOutput]:
        """Return sampled tactics and value estimates for a state batch."""
        if not observations:
            return []
        if not self._inference_models:
            raise RuntimeError('Sampling requires an RL configuration.')

        self.eval()
        encoded = self.tokenizer(
            observations,
            max_length=self.max_state_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        self._sync_inference_replicas()
        input_id_shards = encoded.input_ids.chunk(len(self._inference_models))
        attention_mask_shards = encoded.attention_mask.chunk(
            len(self._inference_models)
        )
        inference_models = self._inference_models[:len(input_id_shards)]
        devices = [
            next(inference_model.parameters()).device
            for inference_model in inference_models
        ]
        inputs = [
            (
                input_ids.to(device),
                attention_mask.to(device),
            )
            for input_ids, attention_mask, device in zip(
                input_id_shards,
                attention_mask_shards,
                devices,
            )
        ]
        for inference_model in inference_models:
            inference_model.eval()
        if len(inference_models) == 1:
            shard_outputs = [inference_models[0](*inputs[0])]
        else:
            shard_outputs = parallel_apply(
                inference_models,
                inputs,
                devices=devices,
            )

        outputs = []
        for value_logits, generated_sequences, logprobs in shard_outputs:
            value_probs = torch.softmax(value_logits, dim=-1)
            values = (
                value_probs * self.value_bins.to(value_logits.device)
            ).sum(dim=-1).tolist()
            generated_actions = self.tokenizer.batch_decode(
                generated_sequences,
                skip_special_tokens=True,
            )
            shard_logprobs = logprobs.tolist()
            for observation_index, value in enumerate(values):
                start = observation_index * self.num_sampled_actions
                end = start + self.num_sampled_actions
                action_logprobs: Dict[Action, float] = {}
                for action, logprob in zip(
                    generated_actions[start:end],
                    shard_logprobs[start:end],
                ):
                    action_logprobs[action] = max(
                        logprob,
                        action_logprobs.get(action, float('-inf')),
                    )
                outputs.append(NetworkSamplingOutput(
                    action_logprobs=action_logprobs,
                    value=value,
                ))

        return outputs

    def update(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, float]]
    ) -> float:
        """Apply one optimizer update from a replay batch."""
        self.train()
        loss = self._loss_fn(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._inference_replicas_stale = True
        return loss.detach().item()

    def evaluate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, float]]
    ) -> float:
        """Evaluate the combined policy and value loss."""
        self.eval()
        with torch.no_grad():
            return self._loss_fn(batch).item()

    def _ensure_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension to a single token sequence."""
        if tokens.dim() == 1:
            return tokens.unsqueeze(0)
        return tokens

    def _sync_inference_replicas(self) -> None:
        """Refresh inference replicas after the primary model changes."""
        if not self._inference_replicas_stale:
            return
        primary_state = self._inference_models[0].state_dict()
        for inference_model in self._inference_models[1:]:
            inference_model.load_state_dict(primary_state)
        self._inference_replicas_stale = False

    def _mean_pool_encoder_state(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean-pool encoder states over non-padding tokens."""
        if attention_mask is None:
            return hidden_state.mean(dim=1)

        mask = attention_mask.to(
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        ).unsqueeze(-1)
        return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
