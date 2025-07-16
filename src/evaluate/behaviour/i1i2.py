import itertools
import logging
from collections import Counter

import numpy as np
import scipy.stats
import xarray
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr

from brainio.assemblies import walk_coords, DataAssembly
from brainscore_core.metrics import Metric, Score
from brainscore_vision.metric_helpers.transformations import apply_aggregate
from brainscore_vision.utils import fullname

from brainscore_vision.metrics.i1i2.metric import _O, I2n

class O2_all_choices(_O):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, collapse_distractors=False, normalize=False, repetitions=10, **kwargs)

    # this modification treats each trial as response to all choices, instead of 2AFC
    def build_response_matrix_from_responses(self, responses):
        num_choices = [(image_id, choice) for image_id, choice in
                        zip(responses['stimulus_id'].values, responses.values)]
        num_choices = Counter(num_choices)
        choices = np.unique(responses)
        num_objects = [[(image_id, obj) for obj in choices] for image_id in responses['stimulus_id'].values]
        num_objects = Counter(itertools.chain(*num_objects))

        image_ids, indices = np.unique(responses['stimulus_id'], return_index=True)
        truths = responses['truth'].values[indices]
        image_dim = responses['stimulus_id'].dims
        coords = {**{coord: (dims, value) for coord, dims, value in walk_coords(responses)},
                    **{'choice': ('choice', choices)}}
        coords = {coord: (dims, value if dims != image_dim else value[indices])  # align image_dim coords with indices
                    for coord, (dims, value) in coords.items()}
        response_matrix = np.zeros((len(image_ids), len(choices)))
        for (image_index, image_id), (choice_index, choice) in itertools.product(
                enumerate(image_ids), enumerate(choices)):
            if truths[image_index] == choice:  # object == choice, ignore
                p = np.nan
            else:
                # divide by number of times where object was one of the two choices (target or distractor)
                p = (num_choices[(image_id, choice)] / num_objects[(image_id, choice)]) \
                    if num_objects[(image_id, choice)] > 0 else np.nan
            response_matrix[image_index, choice_index] = p
        response_matrix = DataAssembly(response_matrix, coords=coords, dims=responses.dims + ('choice',))
        return response_matrix

    @classmethod
    def correlate(cls, source_response_matrix, target_response_matrix, skipna=False, collapse_distractors=False):
        # align
        if collapse_distractors:
            source_response_matrix = source_response_matrix.sortby('task_right')
            target_response_matrix = target_response_matrix.sortby('task_right')
        else:
            source_response_matrix = source_response_matrix.sortby('task_right').sortby('task_left')
            target_response_matrix = target_response_matrix.sortby('task_right').sortby('task_left')
            assert all(source_response_matrix['task_left'].values == target_response_matrix['task_left'].values)
        assert all(source_response_matrix['task_right'].values == target_response_matrix['task_right'].values)
        # flatten and mask out NaNs
        source, target = source_response_matrix.values.flatten(), target_response_matrix.values.flatten()
        non_nan = ~np.isnan(target)
        non_nan = np.logical_and(non_nan, (~np.isnan(source) if skipna else 1))
        source, target = source[non_nan], target[non_nan]
        # assert not any(np.isnan(source))
        correlation, p = pearsonr(source, target)
        return correlation

    # # we stratify by truth and choice when building halves
    # def generate_halves(self, assembly, random_state):
    #     truths = assembly['truth'].values
    #     choices = assembly.values
    #     strats = np.array([f"{t}-{c}" for t,c in zip(truths, choices)])
    #     half1, half2 = [], []
    #     for s in np.unique(strats, axis=0):
    #         indices = np.where(strats == s)[0]
    #         random_state.shuffle(indices)
    #         half1.extend(indices[:int(len(indices) / 2)])
    #         half2.extend(indices[int(len(indices) / 2):])
    #     halves = assembly[half1], assembly[half2]
    #     return halves

class _get_raw:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, *args, **kwargs):
        ret = self.metric(*args, **kwargs)
        ret = ret.raw.transpose(..., "split").values  # [..., n_splits]
        return ret

    def ceiling(self, data, **kwargs):
        ceil = self.metric.ceiling(data, **kwargs)
        return ceil.raw.transpose(..., "split")

metrics = {
    "O2": _get_raw(O2_all_choices()),
    "I2n": _get_raw(I2n()),
}