import numpy as np
import xarray as xr
from tqdm import tqdm
from .utils import pearsonr, take_layers, safe_parallel, pack_multi_np_rets, copy_random_state, split, align_stimulus_paths


def evaluate_per_layer(activation, assembly, layers, metric, mt_benchmark=False, **kwargs):
    from joblib import Parallel, delayed
    evaluate_metric_ = evaluate_metric if not mt_benchmark else evaluate_metric_mt_benchmark
    activations = [take_layers(activation, [layer]) for layer in layers]
    rets = safe_parallel(evaluate_metric_, [(activation_, assembly, metric) for activation_ in activations], kwargs)
    rets = pack_multi_np_rets(rets)  # [num_layers, num_neuroids, n_splits]
    return rets

def evaluate_metric(activation, assembly, metric, n_splits=10, train_valid_test_ratios=(0.8, 0.1, 0.1), random_seed=62):
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
    assert np.all(activation.time_bin.values == assembly.time_bin.values)
    activation = activation.transpose('stimulus_path', 'time_bin', 'neuroid').values
    assembly = assembly.transpose('stimulus_path', 'time_bin', 'neuroid').values

    ns = activation.shape[-1], assembly.shape[-1]
    for n, rs in enumerate(tqdm(random_states)):
        data_splits = split([activation, assembly], train_valid_test_ratios, rs)
        data_splits = [[d.reshape(-1, n) for d, n in zip(data_split, ns)] for data_split in data_splits]

        test_scores = metric(*data_splits)
        if not all_test_scores:
            all_test_scores = [[] for _ in test_scores]
        for i, test_score in enumerate(test_scores):
            all_test_scores[i].append(test_score)
    
    for i in range(len(all_test_scores)):
        all_test_scores[i] = np.stack(all_test_scores[i], axis=-1)
    return all_test_scores  # [n_neuroids, n_splits]


def evaluate_metric_mt_benchmark(activation, assembly, metric, clip_duration=5000, n_splits=10, train_valid_test_ratios=(0.8, 0.1, 0.1), random_seed=62):
    random_states = [np.random.RandomState(random_seed + i) for i in range(n_splits)]
    activation, assembly = align_stimulus_paths(activation, assembly)
    assert np.all(activation.time_bin.values == assembly.time_bin.values)
    time_bin_start = assembly.time_bin_start.values[0]
    time_bin_end = assembly.time_bin_end.values[0]
    time_interval = time_bin_end - time_bin_start
    num_bins = int(clip_duration / time_interval)
    assert time_interval * num_bins == clip_duration

    activation = activation.transpose('stimulus_path', 'time_bin', 'neuroid').values
    assembly = assembly.transpose('stimulus_path', 'time_bin', 'neuroid').values

    n_acti = activation.shape[-1]
    n_asm = assembly.shape[-1]
    n_pres = activation.shape[0]
    n_time = activation.shape[1]
    ret_valid = [None] * n_asm
    ret_test = [None] * n_asm
    for s in range(n_pres):
        asm = assembly[s]
        acti = activation[s]
        neuroid_mask = ~np.isnan(asm).all(axis=0)
        neuroid_id = np.arange(n_asm)[neuroid_mask].item()
        asm = asm[:, [neuroid_id]].reshape(-1, 1)

        acti = acti.reshape(n_time//num_bins, num_bins, n_acti)
        asm = asm.reshape(n_time//num_bins, num_bins, 1)
        ns = n_acti, 1

        all_test_scores = []
        for _, rs in enumerate(tqdm(random_states)):
            data_splits = split([acti, asm], train_valid_test_ratios, rs)
            data_splits = [[d.reshape(-1, n) for d, n in zip(data_split, ns)] for data_split in data_splits]
            
            test_scores = metric(*data_splits)
            if not all_test_scores:
                all_test_scores = [[] for _ in test_scores]
            for i, test_score in enumerate(test_scores):
                all_test_scores[i].append(test_score)
        
        ret_valid[neuroid_id] = np.stack(all_test_scores[0], axis=-1)
        ret_test[neuroid_id] = np.stack(all_test_scores[1], axis=-1)
    
    ret_valid = np.concatenate(ret_valid, axis=0)
    ret_test = np.concatenate(ret_test, axis=0)
    return ret_valid, ret_test  # [n_neuroids, n_splits]