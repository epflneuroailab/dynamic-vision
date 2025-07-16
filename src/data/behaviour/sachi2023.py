import os
import pandas as pd
import random
import numpy as np
import xarray as xr

from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet
from ...evaluate.behaviour.i1i2 import metrics

path = "/home/ytang/workspace/data/oleo/behavioral/turk_experiment_100-v2-catch0.8.nc"  # 0.66 +- 0.09
# path = "/home/ytang/workspace/data/oleo/behavioral/turk_experiment_first_frame_500ms-v2-catch0.8.nc"  # 0.62 +- 0.15

data = xr.open_dataarray(path)
o2 = metrics["O2"]
# o2 = O2()
print(o2.ceiling(data, skipna=True))
breakpoint()

MAP_TO_KINETICS = {
    'Hula hoop': 'hula_hooping',
    'Hurling': 'hurling_sport',
    'Paintball': '=',
    'Powerbocking': 'x',
    'Rollerbalding': 'roller_skating',
}