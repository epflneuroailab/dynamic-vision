import numpy as np
import xarray as xr
from tqdm import tqdm
from .utils import pearsonr, take_layers, safe_parallel, pack_multi_np_rets, copy_random_state, split, align_stimulus_paths


def evaluate_per_layer(activation, assembly, layers, metric, **kwargs):
    from joblib import Parallel, delayed
    activations = [take_layers(activation, [layer]) for layer in layers]
    stratified_vals = _get_stratified_vals(assembly)
    rets = safe_parallel(evaluate_metric, [(activation_, assembly, metric, stratified_vals) for activation_ in activations], kwargs)
    rets = pack_multi_np_rets(rets)  # [num_layers, num_neuroids, n_splits]
    return rets

def evaluate_metric(activation, assembly, metric, stratified_vals, n_splits=10, train_valid_test_ratios=(0.8, 0.1, 0.1), random_seed=52):
    """
    Evaluate the metric on the activation and assembly with 'n_splits' splits.

    Parameters
    ----------
    activation : xarray.DataArray
    assembly : xarray.DataArray
    metric : function
        metric((X_train, Y_train), *(X_test, Y_test)) -> *test_scores [num_neuroids]
    """

    random_states = [np.random.RandomState(random_seed + i) for i in range(n_splits)]
    all_test_scores = []
    activation, assembly = align_stimulus_paths(activation, assembly)
    activation = activation.transpose('stimulus_path', 'neuroid').values
    assembly = assembly.transpose('stimulus_path', 'label').values
    for n, rs in enumerate(tqdm(random_states)):
        data_splits = split([activation, assembly], train_valid_test_ratios, rs, stratified_vals=stratified_vals)
        test_scores = metric(*data_splits)
        if not all_test_scores:
            all_test_scores = [[] for _ in test_scores]
        for i, test_score in enumerate(test_scores):
            all_test_scores[i].append(test_score)
    
    for i in range(len(all_test_scores)):
        all_test_scores[i] = np.stack(all_test_scores[i], axis=-1)
    return all_test_scores  # [n_neuroids, n_splits]

def _get_stratified_vals(assembly):
    coords = []
    
    # add labels
    if assembly.sizes['label'] == 1:
        coords.append(assembly.values.flatten())
    else:
        return

    # add coherence if exists <- ding2012
    if 'coherence' in assembly.coords:
        coords.append(assembly['coherence'].values)

    # reshape
    coords = list(zip(*coords))
    return coords