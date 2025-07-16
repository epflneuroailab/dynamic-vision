import numpy as np
from brainio.stimuli import StimulusSet
from brainio.assemblies import DataAssembly

from brainscore_vision import load_dataset as bs_load_dataset
from brainscore_vision import load_stimulus_set as bs_load_stimulus_set


def load_dataset():
    # dataset = bs_load_dataset('MajajHong2015.public')
    stimulus_set = bs_load_stimulus_set('hvm-public')
    properties = ['s', 'rxy', 'ryz', 'rxz', 'tz', 'ty']
    data = stimulus_set[properties].values
    stimulus_ids = stimulus_set.stimulus_ids.values

    assembly = DataAssembly(
        data,
        dims=('stimulus_id', 'label'),
        coords={
            'stimulus_id': ('stimulus_id', stimulus_ids),
            'label': ('label', properties),
        }
    )

    assembly = assembly.stack(presentation=("stimulus_id", ))

    # attach stimuluset 
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier

    return assembly