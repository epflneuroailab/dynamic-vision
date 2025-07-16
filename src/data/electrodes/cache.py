import os
import pickle
from .. import data_store

from . import majajhong2015, freemanziemba2013, crcns_pvc1, crcns_mt2, oleo2023

# # majajhong2015
# for name in ['majajhong2015-V4', 'majajhong2015-IT']:
#     print(f"Saving {name} to cache...")
#     data = majajhong2015.load_dataset(name)
#     data_store.store(data, name)

# # freemanziemba2013
# for name in ['freemanziemba2013-V1', 'freemanziemba2013-V2']:
#     print(f"Saving {name} to cache...")
#     data = freemanziemba2013.load_dataset(name)
#     data_store.store(data, name)

# # crcns-pvc1
# name = 'crcns-pvc1'
# print(f"Saving {name} to cache...")
# data = crcns_pvc1.load_dataset(name)
# data_store.store(data, name)

# # crcns-mt2
# name = 'crcns-mt2'
# print(f"Saving {name} to cache...")
# data = crcns_mt2.load_dataset(name)
# data_store.store(data, name)

# oleo2023
name = 'oleo-hacs250'
print(f"Saving {name} to cache...")
data = oleo2023.load_dataset('oleo.hacs250-pclips-temporal')
data_store.store(data, name)