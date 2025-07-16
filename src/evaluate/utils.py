import scipy
import numpy as np
import xarray as xr
from functools import partial
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from brainio.assemblies import BehavioralAssembly
from joblib import Parallel, delayed
from tqdm import tqdm
from .hrf import convolve_hrf_assembly


NUM_PROCESSES = -1
if NUM_PROCESSES == 1:
    print("==============================================================================")
    print("Warning: NUM_PROCESSES is set to 1, this may slow down the evaluation process.")
    print("==============================================================================")

def align_stimulus_paths(*assemblies, split_coord="stimulus_path"):
    ret = []
    for assembly in assemblies:
        stimulus_paths = get_coord(assembly, split_coord)
        sorted_indices = np.argsort(stimulus_paths)
        stimulus_paths = [str(p) for p in stimulus_paths[sorted_indices]]
        asm = assembly.isel(stimulus_path=sorted_indices)
        target_coord = "stimulus_path" if "stimulus_path_level_0" not in str(asm) else "stimulus_path_level_0"  # xarray is just stupid
        asm = asm.reset_index("stimulus_path").assign_coords({target_coord: ("stimulus_path", stimulus_paths)})
        ret.append(asm)
    
    stimulus_paths = get_coord(ret[0], split_coord)
    for assembly in ret[1:]:
        if not np.array_equal(stimulus_paths, get_coord(assembly, split_coord)):
            raise ValueError("Stimulus paths are not aligned.")

    return ret

def _split(array, ratios, random_state, stratified_vals=None):
    if stratified_vals is not None:
        # make groups based on stratified_vals
        vals = [hash(str(val)) for val in stratified_vals]
        groups = {}
        for i, val in enumerate(vals): groups.setdefault(val, []).append(i)
        subarrays = [array[ids] for ids in groups.values()]
        rs = copy_random_state(random_state)
        subrets = [_split(arr, ratios, rs) for arr in subarrays]
        rets = list(zip(*subrets))
        rets = [np.concatenate(ret) for ret in rets]
        return rets

    indices = list(range(len(array)))
    random_state.shuffle(indices)

    rets = []
    start = 0
    for ratio in ratios:
        end = start + int(len(indices) * ratio)
        rets.append(array[indices[start:end]])
        start = end
    
    # pad extra data to the last split
    if end < len(indices):
        to_add = array[indices[start:]]
        rets[-1] = np.concatenate([rets[-1], to_add])

    return rets
    
# return a list of tuples, each tuple contains the splits of the arrays
# [(a_train, b_train), (a_test, b_test)]
def split(arrays, ratios, random_state, stratified_vals=None):
    num_arr = len(arrays)
    rss = [copy_random_state(random_state) for _ in range(num_arr)]
    return list(zip(*[_split(array, ratios, rs, stratified_vals) for array, rs in zip(arrays, rss)]))

def make_stimulus_paths(assemblies, make_label=False):
    ret = {}
    for name, assembly in assemblies.items():
        id_map = assembly.stimulus_set.stimulus_paths
        ids = assembly.stimulus_id.values
        paths = [id_map[id] for id in ids]
        if "presentation" in assembly.dims:
            assembly = assembly.reset_index('presentation').rename(presentation='stimulus_path')\
                                .assign_coords(stimulus_path=paths)
            if make_label:
                labels = assembly.values.flatten()
                assembly = assembly.assign_coords(decoding_label=('stimulus_path', labels))
        else:
            assembly = assembly.rename({"stimulus_id": "stimulus_path"})
            assembly = assembly.assign_coords(stimulus_path=paths)
        ret[name] = assembly
    return ret

# xarray is simply stupid, this is a monkey patch due to this
def get_coord(asm, split_coord):
    if f"{split_coord}_level_0" in str(asm):
        split_coord = f"{split_coord}_level_0"
    vals = getattr(asm, split_coord).values
    if isinstance(vals[0], (list, tuple)):
        return np.array([val[0] for val in vals])
    return vals

def time_align(activation, assembly):
    # discard the extra time bins in assembly
    time_bin_start = activation.time_bin_start.values[0]
    time_bin_end = activation.time_bin_end.values[-1]
    assembly = assembly.isel(time_bin=assembly.time_bin_start >= time_bin_start)
    assembly = assembly.isel(time_bin=assembly.time_bin_end <= time_bin_end)
    return assembly_time_align(activation, assembly.time_bin.values), assembly

def time_delay(x, delay):
    # add time delay and get common time bins   
    cls_ = x.__class__
    x = x.reset_index("time_bin")
    time_bin_starts = x["time_bin_start"].values + delay
    time_bin_ends = x["time_bin_end"].values + delay
    x = x.assign_coords(time_bin_start=("time_bin", time_bin_starts), time_bin_end=("time_bin", time_bin_ends))
    return cls_(x)

def time_hrf(x, tr):
    return convolve_hrf_assembly(x, tr)

def take_layers(data, layers):
    if not isinstance(data, dict):
        return data.isel(neuroid=data.layer.isin(layers))
    else:
        return {benchmark: data[benchmark].isel(neuroid=data[benchmark].layer.isin(layers)) for benchmark in data}

def take_benchmarks(data, benchmarks):
    return {benchmark: data[benchmark] for benchmark in benchmarks}

def make_fitting_assembly(fitting_stimulus_set):
    if "truth" not in fitting_stimulus_set:
        truths = fitting_stimulus_set["image_label"].values.flatten()
    else:
        truths = fitting_stimulus_set["truth"].values.flatten()
    ids = fitting_stimulus_set["stimulus_id"].values
    id_map = fitting_stimulus_set.stimulus_paths
    paths = [id_map[id] for id in ids]
    assembly = BehavioralAssembly(
        np.array(truths)[:, None],
        dims=['stimulus_path', 'label'],
        coords={
            "stimulus_path": ("stimulus_path", paths),
            "decoding_label": ("stimulus_path", truths)
        }
    )
    assembly.attrs["stimulus_set"] = fitting_stimulus_set
    return assembly

def pearsonr(x, y):
    # x, y: (n_samples, n_features)
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True) + 1e-6
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True) + 1e-6

    r = ((xm / normxm) * (ym / normym)).sum(axis=0)

    return r

def pack_multi_np_rets(rets):
    rets = list(zip(*rets))
    return [np.stack(ret) for ret in rets]

def copy_random_state(random_state):
    rs = np.random.RandomState()
    rs.set_state(random_state.get_state())
    return rs

class ScaledLogisticRegression:
    def __init__(self, max_iter=3000, C=1.0):
        self._scaler = StandardScaler()
        self._classifier = LogisticRegression(max_iter=max_iter, C=C)

    def _scaler_transform(self, X):
        X = self._scaler.transform(X)
        X[np.isinf(X)] = 0
        return X

    def fit(self, X, Y):
        X = self._scaler.fit_transform(X)
        self._classifier.fit(X, Y)

    def predict(self, X):
        X = self._scaler_transform(X)
        return self._classifier.predict(X)

    def predict_proba(self, X):
        X = self._scaler_transform(X)
        return self._classifier.predict_proba(X)

    def score(self, X, Y):
        X = self._scaler_transform(X)
        return self._classifier.score(X, Y)
    
    @property
    def classes_(self):
        return self._classifier.classes_

def repeat_last_time_bin(xarr):
    # repeat the last time bin to avoid the last-time-bin problem
    last_time_bin = xarr.isel(time_bin=[-1])
    last_time_bin_start = last_time_bin.time_bin_start.values.item()
    last_time_bin_end = last_time_bin.time_bin_end.values.item()
    time_interval = last_time_bin_end - last_time_bin_start
    time_bin_start = last_time_bin_start + time_interval
    time_bin_end = last_time_bin_end + time_interval
    last_time_bin = last_time_bin.reset_index('time_bin').\
        assign_coords(time_bin_start=('time_bin', [time_bin_start]), time_bin_end=('time_bin', [time_bin_end])).\
        set_index(time_bin=['time_bin_start', 'time_bin_end'])
    return xr.concat([xarr, last_time_bin], dim="time_bin")

def safe_parallel(func, args_list, kwargs):
    try:
        from joblib import Parallel, delayed
        rets = Parallel(n_jobs=NUM_PROCESSES)(delayed(func)(*args, **kwargs) 
                        for args in args_list)
    except:
        rets = [func(*args, **kwargs) for args in args_list]
    return rets