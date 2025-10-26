import os


# Detect if .env file exists and load it
home_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(home_path, '.env')
if os.path.exists(env_path):
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)

CACHE_HOME = os.getenv('CACHE_HOME', os.path.dirname(os.path.abspath(__file__)) + '/cache')

defaults = {
    "RESULTCACHING_HOME": f"{CACHE_HOME}/.resultcaching",
    "MMAP_HOME": f"{CACHE_HOME}/.mmap",
    "BRAINIO_HOME": f"{CACHE_HOME}/.brainio2",
    "BRAINSCORE_HOME": f"{CACHE_HOME}/.brain-score",
    "TORCH_HOME": f"{CACHE_HOME}/.torch",
    "HF_HOME": f"{CACHE_HOME}/.hf",
    "STORE_HOME": f"{CACHE_HOME}/.store",
    "NO_ALBUMENTATIONS_UPDATE": "1",
    "RESULTCACHING_DISABLE": '0',
}

for key, value in defaults.items():
    if key not in os.environ:
        os.environ[key] = value


import warnings

# Suppress all future warnings
warnings.filterwarnings("ignore")