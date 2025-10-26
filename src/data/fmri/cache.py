import os
import pickle
from .. import data_store
from brainscore_vision import load_dataset


for (name, dataset_id) in [
    ("savasegal2023-fmri-defeat", 'Savageal2023-fMRI-Defeat'),
    ("savasegal2023-fmri-growth", 'Savageal2023-fMRI-Growth'),
    ("savasegal2023-fmri-iteration", 'Savageal2023-fMRI-Iteration'),
    ("savasegal2023-fmri-lemonade", 'Savageal2023-fMRI-Lemonade'),
    ("keles2024-fmri", 'Keles2024-FMRI'),
    ("berezutskaya2021-fmri", 'Berezutskaya2021-FMRI'),
    ("lahner2024-fmri", "Lahner2024-fMRI"),
    ("mcmahon2023-fmri", "McMahon2023-fMRI"),
]:
    if data_store.exists(name):
        print(f"Dataset {name} already exists in cache, skipping...")
    else:
        print(f"Saving {name} to cache...")
        # load from brainscore
        data = load_dataset(dataset_id)
        data_store.store(data, name)