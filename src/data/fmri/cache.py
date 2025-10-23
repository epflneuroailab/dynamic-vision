import os
import pickle
from .. import data_store

def _fake_load_from_brainscore(name):
    import sys
    sys.path.insert(0, '/mnt/scratch/ytang/migrate')
    from datasets import load_data
    data = load_data(name)
    for k, v in data.stimulus_set.stimulus_paths.items():
        data.stimulus_set.stimulus_paths[k] = v.replace('/upschrimpf2', '')
    return data

for name in [
    "savasegal2023-fmri-defeat",
    "savasegal2023-fmri-growth",
    "savasegal2023-fmri-iteration",
    "savasegal2023-fmri-lemonade",
    "keles2024-fmri",
    "berezutskaya2021-fmri",
    "lahner2024-fmri",
    "mcmahon2023-fmri",
]:
    # if data_store.exists(name):
    #     print(f"Dataset {name} already exists in cache, skipping...")
    # else:
        print(f"Saving {name} to cache...")
        # load from brainscore
        data = _fake_load_from_brainscore(name)
        data_store.store(data, name)