import numpy as np
import xarray as xr
from tqdm import tqdm
from .utils import pearsonr, take_layers, safe_parallel, pack_multi_np_rets, copy_random_state, split


def evaluate_per_layer(activations, assemblies, layers, clip_duration, metric, **kwargs):
    from joblib import Parallel, delayed
    assemblies = _normalize_assemblies(assemblies)
    activations_ = [take_layers(activations, [layer]) for layer in layers]
    rets = safe_parallel(evaluate_metric, [(activation_, assemblies, clip_duration, metric) for activation_ in activations_], kwargs)
    rets = pack_multi_np_rets(rets)  # [num_layers, num_neuroids, n_splits]
    return rets
    
def evaluate_metric(activations, assemblies, clip_duration, metric, n_splits=10, train_valid_test_ratios=(0.8, 0.1, 0.1), kfold=5, random_seed=42):
    """
    Evaluate the metric on the activations and assemblies with 'n_splits' splits.

    Parameters
    ----------
    activations : dict
        A dictionary of xarray.DataArray, where the key is the benchmark name and the value is the activations.
    assemblies : dict
        A dictionary of xarray.DataArray, where the key is the benchmark name and the value is the assemblies.
    metric : function
        metric((X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)) -> valid_scores, test_scores [num_neuroids]
    clip_duration : int
        The duration of each clip in seconds.
    """

    random_states = [np.random.RandomState(random_seed + i) for i in range(n_splits)]
    valid_scores = []
    test_scores = []
    for n, rs in enumerate(tqdm(random_states)):
        activations_, assemblies_ = _compile(activations, assemblies, rs)
        X_train, X_valid, X_test = _time_contiguous_split(activations_, clip_duration, train_valid_test_ratios, copy_random_state(rs))
        Y_train, Y_valid, Y_test = _time_contiguous_split(assemblies_, clip_duration, train_valid_test_ratios, copy_random_state(rs))
        valid_score, test_score = metric((X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test))
        valid_scores.append(valid_score)
        test_scores.append(test_score)
    
    valid_scores = np.stack(valid_scores, axis=-1)
    test_scores = np.stack(test_scores, axis=-1)
    return valid_scores, test_scores  # [n_neuroids, n_splits]

def _compile(activations, assemblies, random_state):
    """
    This function compile the activations and assemblies from different benchmarks into a single dataset.
    The dataset is of shape [N x D], where N is the number of time points from all datasets. This dimension
    is contiguous in time. D is the number of neurons in activations or assemblies. The compilation should
    randomly shuffle the order of the datasets.
    
    The assemblies in dataset "McMahon2023" will be replicated for 3 times, since the 3 second videos
    in their dataset only contain the averaged single fMRI signal for each video. 

    The compiled data will look like this:
    | ----- movie1 ----- | clip1 | clip2 | ... | ----- movie2 ----- | clip4 | ... | ----- movieN ----- |
    """
    _activations = {}
    _assemblies = {}
    stimulus_paths = []
    for (benchmark, activation), assembly in zip(activations.items(), assemblies.values()):
        paths = activation.stimulus_path.values
        stimulus_paths.extend([*paths])
        assert (activation.stimulus_path == assembly.stimulus_path).all()
        if benchmark in ["mcmahon2023-fmri", "lahner2024-fmri"]:
            activation, assembly = _expand_time_bins_mcmahon2023(activation, assembly)
        for stimulus_path in paths:
            _activations[stimulus_path] = activation.sel(stimulus_path=stimulus_path).transpose("time_bin", "neuroid").values
            _assemblies[stimulus_path] = assembly.sel(stimulus_path=stimulus_path).transpose("time_bin", "neuroid").values
    
    random_state.shuffle(stimulus_paths)
    activations = np.concatenate([_activations[stimulus_path] for stimulus_path in stimulus_paths])
    assemblies = np.concatenate([_assemblies[stimulus_path] for stimulus_path in stimulus_paths])
    return activations, assemblies

def _time_contiguous_split(compliation, chunk_duration, ratios, random_state):
    """
    This function split the data produced by compile function into chunks of duration `chunk_duration` seconds.
    The chunks are regrouped into different datasets according to the ratios.
    """

    chunk_duration = int(chunk_duration)
    chunks = [compliation[i:i+chunk_duration] for i in range(0, len(compliation), chunk_duration)]
    chunks = np.array(chunks, dtype=object)
    ret = split([chunks], ratios, random_state)
    ret = [np.concatenate(r[0]) for r in ret]
    return ret

def _normalize_assemblies(assemblies):
    """
    This function normalize the assemblies, so that they can be compiled to be the targets in a single regression.
    """

    def _normalize(assembly):
        mean = assembly.mean(["time_bin", "stimulus_path"])
        std = assembly.std(["time_bin", "stimulus_path"]) + 1e-6
        return (assembly - mean) / std

    return {benchmark: _normalize(assembly) for benchmark, assembly in assemblies.items()}

def _expand_time_bins_mcmahon2023(activation, assembly):
    def _repeat_time_bins(xarr):
        cls_ = xarr.__class__
        xarr = xr.concat([xarr] * 3, dim="time_bin")
        xarr = xarr.reset_index("time_bin")
        time_bin_starts = [0, 1000, 2000]
        time_bin_ends = [1000, 2000, 3000]
        xarr = xarr.assign_coords(time_bin_start=("time_bin", time_bin_starts), time_bin_end=("time_bin", time_bin_ends))
        xarr = cls_(xarr)
        return xarr

    return _repeat_time_bins(activation), _repeat_time_bins(assembly)
