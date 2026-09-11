import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'models'
MODEL_ALIASES = {
    'qwen3.6-27b': 'Qwen/Qwen3.6-27B',
    'qwen3.5-9b': 'Qwen/Qwen3.5-9B',
}


def model_dir_name(model_name: str) -> str:
    """Return a filesystem-friendly directory name for a Hugging Face model."""
    return model_name.replace('/', '--')


def download_model(model_name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Download a model and tokenizer from Hugging Face into models_dir."""
    model_name = MODEL_ALIASES.get(model_name.lower(), model_name)
    output_dir = models_dir / model_dir_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=output_dir,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download a Hugging Face model and tokenizer.'
    )
    parser.add_argument(
        'model_name',
        help=(
            'Hugging Face model name, or alias qwen3.6-27b or qwen3.5-9b.'
        ),
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=MODELS_DIR,
        help='Directory where downloaded models are stored.',
    )
    return parser.parse_args()


def main():
    """Download the requested model and print the local path."""
    args = parse_args()
    output_dir = download_model(args.model_name, args.models_dir)
    print(f'Downloaded {args.model_name} to {output_dir}')


if __name__ == '__main__':
    main()
