import os
import pickle
from .. import data_store

from . import imagenet2012, afd2022, kinetics400, smthsmthv2, ding2012, vggface2, hdm05, selfmotion, mcmahon2023, majajhong2015, cueconflict, mtool


# # imagenet2012
# name = 'imagenet2012'
# print(f"Saving {name} to cache...")
# data = imagenet2012.load_dataset()
# data_store.store(data, name)

# # afd2022
# name = 'afd2022'
# print(f"Saving {name} to cache...")
# data = afd2022.load_dataset()
# data_store.store(data, name)

# # kinetics400
# for name in ['kinetics400', 'kinetics400-static']:
#     print(f"Saving {name} to cache...")
#     data = kinetics400.load_dataset(name)
#     data_store.store(data, name)

# # smthsmthv2
# name = 'smthsmthv2'
# print(f"Saving {name} to cache...")
# data = smthsmthv2.load_dataset()
# data_store.store(data, name)

# # ding2012
# name = 'ding2012'
# print(f"Saving {name} to cache...")
# data = ding2012.load_dataset()
# data_store.store(data, name)

# # vggface2
# name = 'vggface2'
# print(f"Saving {name} to cache...")
# data = vggface2.load_dataset(name)
# data_store.store(data, name)

# # hdm05
# name = 'hdm05'
# print(f"Saving {name} to cache...")
# data = hdm05.load_dataset()
# data_store.store(data, name)

# name = 'selfmotion'
# print(f"Saving {name} to cache...")
# data = selfmotion.load_dataset()
# data_store.store(data, name)

# name = 'mcmahon2023-social'
# print(f"Saving {name} to cache...")
# data = mcmahon2023.load_dataset()
# data_store.store(data, name)

# name = 'majajhong2015-pose'
# print(f"Saving {name} to cache...")
# data = majajhong2015.load_dataset()
# data_store.store(data, name)

# name = 'cueconflict'
# print(f"Saving {name} to cache...")
# data = cueconflict.load_dataset()
# data_store.store(data, name)

name = 'mechanical_tools'
print(f"Saving {name} to cache...")
data = mtool.load_dataset()
data_store.store(data, name)