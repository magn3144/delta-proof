from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
DATASET_DIR = DATA_DIR / 'dataset'
RUNS_DIR = DATA_DIR / 'runs'
LEAN_PROJECT_DIR = PROJECT_ROOT / 'lean_project'
MODELS_DIR = DATA_DIR / 'models'
NUMINA_MATH_LEAN_PATH = DATASET_DIR / 'numina_math_lean.parquet'
CLEANED_NUMINA_MATH_LEAN_PATH = DATASET_DIR / 'numina_math_lean_cleaned.jsonl'
DEFAULT_THEOREMS_DIR = DATASET_DIR / 'numina_math_lean_passing'
