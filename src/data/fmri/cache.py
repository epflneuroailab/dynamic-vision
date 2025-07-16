import os
import pickle
from .. import data_store

from . import savasegal2023, keles2024, berezutskaya2021, mcmahon2023, lahner2024


# savasegal2023
for movie in ["Defeat.mp4", "Growth.mp4", "Iteration.mp4", "Lemonade.mp4"]:
    name = f'SavaSegal2023-fMRI-{movie}'[:-4].lower()
    print(f"Saving {name} to cache...")
    data = savasegal2023.load_dataset(movie)
    data_store.store(data, name)

# keles2024
name = 'keles2024-fmri'
print(f"Saving {name} to cache...")
data = keles2024.load_dataset()
data_store.store(data, name)

# berezutskaya2021
name = 'berezutskaya2021-fmri'
print(f"Saving {name} to cache...")
data = berezutskaya2021.load_dataset()
data_store.store(data, name)

# mcmahon2023
name = 'mcmahon2023-fmri'
print(f"Saving {name} to cache...")
data = mcmahon2023.load_dataset()
data_store.store(data, name)

# lahner2024
name = 'lahner2024-fmri'
print(f"Saving {name} to cache...")
data = lahner2024.load_dataset()
data_store.store(data, name)