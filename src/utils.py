import itertools
import numpy as np
import xarray as xr
import scipy
from brainio.assemblies import NeuroidAssembly, walk_coords, DataAssembly
from brainscore_vision.benchmark_helpers.neural_common import average_repetition as bs_average_repetition
from brainscore_vision import Score

def combine_splits(*assemblies):
    # combine the split assemblies into a single assembly
    cls = assemblies[0].__class__
    return cls(xr.concat(assemblies, dim='split'))

def split_time(*assemblies, split_ratio=0.5):
    # split the assembly into two parts, based on the split_ratio to the left
    # always keep the first part consecutive in time
    return [_split_time(assembly, split_ratio) for assembly in assemblies]

def _split_time(assembly, split_ratio):
    num_time_bins = len(assembly.time_bin.values)
    split_size = int(num_time_bins*split_ratio)
    split_idx = np.random.randint(0, num_time_bins-split_size)
    split_mask = np.zeros(num_time_bins, dtype=bool)
    split_mask[split_idx:split_idx+split_size] = True
    split = assembly.isel(time_bin=split_mask)
    remain = assembly.isel(time_bin=~split_mask)
    return split, remain

def _check_consecutive(time_bin_starts, time_bin_ends):
    assert (time_bin_starts[1:]==time_bin_ends[:-1]).all()

def _get_common_time(source, target):
    source_time = set([v for v in source.time_bin.values])
    target_time = set([v for v in target.time_bin.values])
    common_time = list(source_time.intersection(target_time))
    common_time.sort(key=lambda x: x[0])
    common_time = np.array(common_time)
    _check_consecutive(common_time[:, 0], common_time[:, 1])
    return common_time

# fetch the source and target where they have a consecutive common time period
def fetch_common_time_bins(source, target):
    source_cls = source.__class__
    target_cls = target.__class__
    common_time = _get_common_time(source, target)

    _check_consecutive(source.time_bin_start.values, source.time_bin_end.values)
    _check_consecutive(target.time_bin_start.values, target.time_bin_end.values)
    common_source = assembly_time_align(source, common_time)
    common_target = assembly_time_align(target, common_time)
    common_source = source_cls(common_source)
    common_target = target_cls(common_target)
    return common_source, common_target

def pearsonr(x, y):
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True) + 1e-8
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True) + 1e-8

    r = ((xm / normxm) * (ym / normym)).sum(axis=0)

    return r

class BatchPearsonR:
    def __init__(self, correlation_coord, neuroid_coord):
        self._correlation = pearsonr
        self._correlation_coord = correlation_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target):
        # align
        prediction = prediction.sortby([self._correlation_coord, self._neuroid_coord])
        target = target.sortby([self._correlation_coord, self._neuroid_coord])
        assert np.array(prediction[self._correlation_coord].values == target[self._correlation_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()
        # compute correlation per neuroid
        neuroid_dims = target[self._neuroid_coord].dims
        assert len(neuroid_dims) == 1
        prediction = prediction.transpose(..., *neuroid_dims)
        target = target.transpose(..., *neuroid_dims)
        correlations = self._correlation(prediction.values, target.values)
        # package
        result = Score(correlations,
                       coords={coord: (dims, values)
                               for coord, dims, values in walk_coords(target) if dims == neuroid_dims},
                       dims=neuroid_dims)
        return result

# ops that also applies to attrs (and attrs of attrs), which are xarrays
def recursive_op(*arrs, op=lambda x:x):
    # the attrs structure of each arr must be the same
    val = op(*arrs)
    attrs = arrs[0].attrs
    for attr in attrs:
        attr_val = arrs[0].attrs[attr]
        if isinstance(attr_val, xr.DataArray):
            attr_arrs = [arr.attrs[attr] for arr in arrs]
            attr_val = recursive_op(*attr_arrs, op=op)
        val.attrs[attr] = attr_val
    return val

def apply_over_dims(callable, *asms, dims, njobs=-1):
    asms = [asm.transpose(*dims, ...) for asm in asms]
    sizes = [asms[0].sizes[dim] for dim in dims]

    def apply_helper(sizes, dims, *asms):
        xarr = []
        attrs = {}
        size = sizes[0]
        rsizes = sizes[1:]
        dim = dims[0]
        rdims = dims[1:]

        if len(sizes) == 1:
            # parallel execution on the last applied dimension
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=njobs)(delayed(callable)(*[asm.isel({dim:s}) for asm in asms]) for s in range(size))
        else:
            results = []
            for s in range(size):
                arr = apply_helper(rsizes, rdims, *[asm.isel({dim:s}) for asm in asms])
                results.append(arr)

        for arr in results:
            if arr is not None:
                for k,v in arr.attrs.items():
                    assert isinstance(v, xr.DataArray)
                    attrs.setdefault(k, []).append(v.expand_dims(dim))
                xarr.append(arr)
       
        if not xarr:
            return
        else:
            xarr = xr.concat(xarr, dim=dim)
            attrs = {k: xr.concat(vs, dim=dim) for k,vs in attrs.items()}
            xarr.coords[dim] = asms[0].coords[dim]
            for k,v in attrs.items():
                attrs[k].coords[dim] = asms[0].coords[dim]
                xarr.attrs[k] = attrs[k]
            return xarr

    return apply_helper(sizes, dims, *asms)

def _efficient_mean_time_bin(activation, chunk_size=1, n_jobs=-1, dim='stimulus_path', time_max=None):
    # n_jobs > 1 seems to lead to much slower performance

    import xarray as xr
    from joblib import Parallel, delayed
    N = activation.sizes[dim]
    cls_ = activation.__class__

    def _run(i):
        print(f"Processing chunk {i}/{N}", end="\r")
        activation_chunk = activation.isel({dim: slice(i, i+chunk_size)})
        if time_max is not None:
            time_starts = activation_chunk.time_bin_start.values
            selected = time_starts < time_max
            activation_chunk = activation_chunk.isel(time_bin=selected)
        skipna = np.isnan(activation_chunk.isel(time_bin=-1)).any().item()
        activation_chunk = activation_chunk.mean(dim="time_bin", skipna=skipna)
        return activation_chunk

    all_data = Parallel(n_jobs=n_jobs)(delayed(_run)(i) for i in range(0, N, chunk_size))        
    activation = xr.concat(all_data, dim=dim)
    return cls_(activation)

def split_halves(N):
    elements = list(range(N))
    
    half_size = N // 2
    
    # Generate all subsets of size N/2
    all_combinations = list(itertools.combinations(elements, half_size))

    def _tag(partition):
        return tuple(sorted(partition))
    
    # Iterate over each combination and its complement
    seen = set()  # To avoid duplicate halves
    for subset in all_combinations:
        subset = set(subset)
        complement = set(elements) - subset
        
        # Create sorted tuples to avoid duplicates
        partition = (tuple(sorted(subset)), tuple(sorted(complement)))
        
        p = _tag(partition)
        if p not in seen:
            seen.add(p)
            yield partition

def average_repetition(assembly):
    ret = bs_average_repetition(assembly)

    if "presentation" in ret.dims:
        ret = ret.reset_index("presentation").rename({"presentation": "stimulus_id"})

    return ret