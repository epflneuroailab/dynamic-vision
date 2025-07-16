import os


os.environ["RESULTCACHING_HOME"] = r"/home/ytang/workspace/data/cache/.resultcaching"
os.environ['MMAP_HOME'] = r'/home/ytang/workspace/data/acache/mmap'
os.environ["BRAINIO_HOME"] = r"/home/ytang/workspace/data/cache/.brainio2"
os.environ["BRAINSCORE_HOME"] = r"/home/ytang/workspace/data/cache/.brain-score"
os.environ['TORCH_HOME'] = r'/home/ytang/workspace/data/cache/.torch'
os.environ['HF_HOME'] = r'/home/ytang/workspace/data/cache/.hf'

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["RESULTCACHING_DISABLE"] = '0'


import warnings

# Suppress all future warnings
warnings.filterwarnings("ignore")