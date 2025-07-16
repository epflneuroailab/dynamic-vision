import numpy as np
from scipy.stats import norm

from .utils import combine_splits, BatchPearsonR, average_repetition
from .store import pickle_store
from .data import DATASETS
from .analysis.fdr import false_discovery_control


def ceiler(assembly, splits=10):
    # compute the split-half consistency of the assembly, across time bins
    # and across the cross-validation folds
    time_corrs = []
    for _ in range(splits):
        split_a, split_b = _split_half(assembly)
        split_a = average_repetition(split_a)
        split_b = average_repetition(split_b)
        time_corr = correlation_over_time_and_presentation(split_a, split_b)
        time_corrs.append(time_corr)
    ceiling = combine_splits(*time_corrs)
    return ceiling

def correlation_over_time_and_presentation(source, target):
    # the source and target both have additionally the neuroid dimension
    assert (source.time_bin.values == target.time_bin.values).all()
    assert (source.stimulus_id.values == target.stimulus_id.values).all()
    
    def stack_pres_time(assembly):
        return assembly.reset_index('time_bin').stack(tmp=('stimulus_id', 'time_bin'))
    
    # over time and over presentations
    source = stack_pres_time(source)
    target = stack_pres_time(target)

    correlation = BatchPearsonR('tmp', 'neuroid_id')(source, target)
    return correlation

def _split_half(assembly):
    # split half the assembly along the repetition dimension
    repetition_values = assembly.repetition.values
    repetition_ids = np.unique(repetition_values)
    num_repeat = len(repetition_ids)
    split_size = int(num_repeat/2)
    split_idx = np.random.randint(0, num_repeat-split_size)
    split_mask = np.zeros(len(assembly.presentation.values), dtype=bool)
    split_mask[np.isin(repetition_values, repetition_ids[split_idx:split_idx+split_size])] = True
    split = assembly.isel(presentation=split_mask)
    remain = assembly.isel(presentation=~split_mask)
    return split, remain


def compute_joint_ceiling(ceiling_data):
    # ceiling_data = {ceiling1, ceiling2, ...}
    # ceiling1 = dataset: [ceiling_score, meta]

    prod = 0
    n = 0
    for dataset, (ceiling, meta) in ceiling_data.items():
        c = ceiling.transpose('split', 'neuroid').values
        num_time_pres = meta['time_bin'] * meta['stimulus_id']
        if dataset in ["lahner2024-fmri", "mcmahon2023-fmri"]:
            num_time_pres = num_time_pres * 3  # 3 seconds
        prod += c * num_time_pres
        n += num_time_pres

    joint_ceiling = prod / n

    # spearman-bronw correction
    joint_ceiling = joint_ceiling * 2 / (joint_ceiling + 1)

    return joint_ceiling  # [split, neuroid:20484]


def compute_joint_ceiling_by_codes(fmri_codes, alpha=0.05, threshold=0.4):
    # compute ceiling
    ceil_store = pickle_store.add_node("ceiling")
    meta_store = pickle_store.add_node("fmri_meta")
    fmri_code_all = ''.join(fmri_codes)
    ceilings = {}
    for fmri_code in fmri_code_all:
        dataset = DATASETS[int(fmri_code)]
        ceiling = ceil_store.load(dataset)
        meta = meta_store.load(dataset)
        ceilings[dataset] = (ceiling, meta)

    ceiling = compute_joint_ceiling(ceilings)
    std = ceiling.std(0)
    ceiling = ceiling.mean(0)
    pvals = 1 - norm.cdf(ceiling / std)
    pvals = false_discovery_control(pvals, method='bh')
    ceiling[pvals > alpha] = np.nan
    ceiling[ceiling < threshold] = np.nan
    return ceiling