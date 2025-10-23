import os


# dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = "/mnt/scratch/ytang/migrate"
CACHE_HOME = os.path.join(dir_path, 'cache')

os.environ["RESULTCACHING_HOME"] = f"{CACHE_HOME}/.resultcaching"
os.environ['MMAP_HOME'] = f'{CACHE_HOME}/.mmap'
os.environ["BRAINIO_HOME"] = f"{CACHE_HOME}/.brainio2"
os.environ["BRAINSCORE_HOME"] = f"{CACHE_HOME}/.brain-score"
os.environ['TORCH_HOME'] = f'{CACHE_HOME}/.torch'
os.environ['HF_HOME'] = f'{CACHE_HOME}/.hf'
os.environ["STORE_HOME"] = f"{CACHE_HOME}/.store"

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["RESULTCACHING_DISABLE"] = '0'


import warnings

# Suppress all future warnings
warnings.filterwarnings("ignore")