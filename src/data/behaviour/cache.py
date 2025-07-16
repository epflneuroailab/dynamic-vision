import os
import pickle

from . import ilic2022, rajalingham2018, han2024
from .. import data_store
from ... import config

fitting_stimuli_store = data_store.add_node("fitting_stimuli")

# # ilic2022
# for name in ['ilic2022-afd5', 'ilic2022-ucf5']:
#     print(f"Saving {name} to cache...")
#     data = ilic2022.load_dataset(name)
#     fitting_stimuli = ilic2022.load_stimulus_set(name+'-fitting_stimuli')
#     data_store.store(data, name)
#     fitting_stimuli_store.store(fitting_stimuli, name+'-fitting_stimuli')

# # rajalingham2018
# name = 'rajalingham2018'
# data =  rajalingham2018.load_dataset()
# fitting_stimuli = rajalingham2018.load_stimulus_set()
# data_store.store(data, name)
# fitting_stimuli_store.store(fitting_stimuli, name+'-fitting_stimuli')

# han2024
types = han2024.TYPES_TO_INCLUDE
for type in types:
    name = f'han2024-{type}'
    data =  han2024.load_dataset(name)
    data_store.store(data, name)
    print(f"Saving {name} to cache...")

for type in ["RGB", "J-6P"]:
    name = f'han2024-{type}-fitting_stimuli'
    fitting_stimuli = han2024.load_stimulus_set(name)
    fitting_stimuli_store.store(fitting_stimuli, name)
    print(f"Saving {name} to cache...")