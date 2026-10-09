"""Fail before training if the isolated imports or Blackwell kernels are unusable."""

import sys
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import torch
from vllm import LLM, SamplingParams
from vllm import _custom_ops
from transformers import AutoTokenizer, T5ForConditionalGeneration

import bitsandbytes
import leantree
from alphaproof.core.network import Network
from alphaproof.training import rl_cli, sft

repo = Path(__file__).resolve().parents[3]
assert Path(sys.prefix) == repo / '.venv-ucloud-b200', sys.prefix
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert version('vllm') == '0.10.2'
assert jax.__version__ == '0.5.3'
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert 'sm_100' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
for index in range(torch.cuda.device_count()):
    assert torch.cuda.get_device_capability(index) == (10, 0)
    with torch.cuda.device(index):
        x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
        assert (x @ x).sum().item() == 32768
        # Exercise the compiled vLLM extension without loading model weights.
        output = torch.empty((32, 16), device='cuda', dtype=torch.bfloat16)
        _custom_ops.silu_and_mul(output, x)
        torch.cuda.synchronize()
    print(f'PyTorch CUDA OK: {torch.cuda.get_device_name(index)}')

for device in jax.devices('gpu'):
    with jax.default_device(device):
        x = jnp.ones((32, 32), dtype=jnp.bfloat16)
        result = jax.jit(jnp.matmul)(x, x).block_until_ready()
        assert float(result.sum()) == 32768
    print(f'JAX CUDA OK: {device}')
print(f'B200 imports/kernels OK: Python {sys.version.split()[0]}, '
      f'torch {torch.__version__}, CUDA {torch.version.cuda}, '
      f'vLLM {version("vllm")}, JAX {jax.__version__}')
