import os

class DATASETS:
    IMAGENET1K = 'imagenet'
    AFD101 = 'afd101'

class OUTPUT_DIM:
    IMAGENET1K = 1000
    AFD101 = 101


OUTPUT_DIM_MAP = {
    getattr(DATASETS, dataset_name): getattr(OUTPUT_DIM, dataset_name)
    for dataset_name in dir(DATASETS) if not dataset_name.startswith("__")
}

WANDB_KEY = os.getenv("WANDB_API_KEY", "dd4c069a20277efa2a87f957cb30d405f5b52df3")