import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from scripts.utils import *
from src.store import pickle_store
from src.models.groups import *

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f4ab')
os.makedirs(FIGURE_DIR, exist_ok=True)


# Tasks to test
DEFAULT_TASKS_TO_TEST = [
    ['imagenet2012'],
    ['afd2022'],
    ['imagenet2012', 'afd2022'],
]

# Motion tasks
ALL_MOTION_TASKS = ["afd2022"]
LOW_MOTION_TASKS = ["ding2012", "selfmotion"]

# Benchmarks to avoid
REDUNDANT = ["han2024-RGB-4F", "han2024-J-6P-4F"]

# Plotting configuration
STREAM_CONSIDERED = ["Early", "Ventral", "D.-dorsal", "V.-dorsal"]
ANCHORS = ["V4", "V3A", "LO1"]


def darken_color(hex_color, factor=0.7):
    """Darken a hex color by a given factor"""
    rgb = mcolors.hex2color(hex_color)
    darker_rgb = tuple(max(0, c * factor) for c in rgb)
    return mcolors.to_hex(darker_rgb)


def get_r2_stats(input_scores, brain_scores, n_resamples=200):
    """Compute R² with confidence intervals"""
    _, r2 = resample_analyze(input_scores, brain_scores, n_resamples=n_resamples)
    mean, ci_low, ci_up = r2
    return mean.item(), ci_low.item(), ci_up.item()


def layer_consistency(layer_mapping, split=10):
    """Compute consistency of layer mapping across models"""
    layer_mapping = np.array(list(layer_mapping.values()))
    N = len(layer_mapping)
    consistency = []
    
    for _ in range(split):
        indices = np.random.permutation(N)
        half1 = np.mean(layer_mapping[indices[:N//2]], 0)
        half2 = np.mean(layer_mapping[indices[N//2:]], 0)
        nans = np.isnan(half1) | np.isnan(half2)
        half1, half2 = half1[~nans], half2[~nans]
        c = np.corrcoef(half1, half2)[0, 1]
        # Spearman-Brown correction
        c = 2 * c / (1 + c)
        consistency.append(c)
    
    return np.array(consistency)


# ============================================================================
# COMPUTATION FUNCTIONS
# ============================================================================

def compute_roi_voxel_percentages(args):
    """Compute task relevance percentages for ROIs and voxels"""
    print("Computing ROI and voxel task percentages...")
    
    fmri, task, models = collect(args, types=['fmri', 'task'])
    ceiling = get_ceiling(args)
    
    task_benchmarks = list(task.keys())
    fmri_scores = np.array([fmri[m].mean(0) for m in models])
    task_scores = np.array([[task[t][m].mean(0) for m in models] 
                           for t in task_benchmarks]).T
    
    # ROI results
    roi_results = {}
    for stream, regions in STREAMS.items():
        for region in regions:
            fmri_scores_region = select_and_ceil(fmri_scores, ceiling, region).reshape(-1, 1)
            for tasks in args.tasks_to_test:
                task_indices = [task_benchmarks.index(t) for t in tasks]
                input_scores = task_scores[:, task_indices]
                r2_stats = get_r2_stats(input_scores, fmri_scores_region, args.num_resample)
                roi_results.setdefault(region, []).append(r2_stats)
                print(f"{region} {tasks}: {r2_stats[0]:.4f} ({r2_stats[1]:.4f}, {r2_stats[2]:.4f})")
    
    cache_key = f"cache.tests.paper_plots.f3.perc.{args.clip_duration}"
    pickle_store.store((roi_results), cache_key)
    print(f"Stored to {cache_key}")
    
    return roi_results


def compute_behaviour_percentages(args):
    """Compute behaviour task percentages"""
    print("Computing behaviour task percentages...")
    
    task, beh, models = collect(args, types=['task', 'beh'])
    
    # Behaviour scores
    beh_scores = {
        beh_name: np.array([np.nanmean(beh[beh_name][m][0], 0) for m in models]) 
        for beh_name in beh if beh_name not in REDUNDANT
    }
    
    aver_beh_scores = np.array([val for val in beh_scores.values()]).mean(0)
    
    # Condition-specific scores
    condition_beh_scores = []
    for condition in ["normal", "shuffle"]:
        beh_scores_ = []
        for beh_name in beh_scores:
            if condition == "shuffle" and beh_name.endswith("-S"):
                beh_scores_.append(beh_scores[beh_name])
            if condition == "normal" and (beh_name.endswith("RGB") or beh_name.endswith("6P")):
                beh_scores_.append(beh_scores[beh_name])
        
        beh_score = np.array(beh_scores_).mean(0)
        condition_beh_scores.append(beh_score)
    
    condition_beh_scores.append(aver_beh_scores)
    
    # Task scores
    task_benchmarks = list(task.keys())
    task_scores = np.array([[task[t][m].mean(0) for m in models] 
                           for t in task_benchmarks]).T
    
    # Compute results for each condition
    beh_results = {}
    for beh_score_case, case in zip(condition_beh_scores, ["normal", "shuffle", 'average']):
        for tasks in args.tasks_to_test:
            task_indices = [task_benchmarks.index(t) for t in tasks]
            input_scores = task_scores[:, task_indices]
            
            nans = np.isnan(beh_score_case)
            beh_score_clean = beh_score_case[~nans]
            input_scores_clean = input_scores[~nans, :]
            
            r2_stats = get_r2_stats(input_scores_clean, beh_score_clean, args.num_resample)
            beh_results.setdefault(case, []).append(r2_stats)
            print(f"{case} {tasks}: {r2_stats[0]:.4f} ({r2_stats[1]:.4f}, {r2_stats[2]:.4f})")
    
    cache_key = f"cache.tests.paper_plots.f3.behaviour.{args.clip_duration}"
    pickle_store.store(beh_results, cache_key)
    print(f"Stored to {cache_key}")
    
    return beh_results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def _get_bar_stats(bar_data):
    """Extract mean and error bars from data"""
    mean = bar_data[:, 0]
    yerr = bar_data[:, 1:]
    yerr = np.stack([mean - yerr[:, 0], yerr[:, 1] - mean], 0)
    return mean, yerr


def plot_bar(img, afd, combined, regions, ymin=None, ymax=None):
    """Plot three-bar comparison"""
    width = 1
    space = 1
    
    xs = np.arange(len(regions)) * (width * 2 + space)
    c_m, c_e = _get_bar_stats(combined)
    i_m, i_e = _get_bar_stats(img)
    a_m, a_e = _get_bar_stats(afd)
    
    plt.bar(xs + width * 0.5, c_m, yerr=c_e, label="Combined", width=width * 2, 
           color=MIXED, error_kw=dict(ecolor=darken_color(MIXED), alpha=0.35))
    plt.bar(xs, i_m, yerr=i_e, label="Obj. Rec.", width=width, 
           color=STATIC, error_kw=dict(ecolor=darken_color(STATIC), alpha=0.35))
    plt.bar(xs + width, a_m, yerr=a_e, label="Action Rec. (MO)", width=width, 
           color=DYNAMIC, error_kw=dict(ecolor=darken_color(DYNAMIC), alpha=0.35))
    
    plt.xlim(-width / 2, len(regions) * (width * 2 + space) - width / 2 - space)
    plt.xticks(np.arange(len(regions)) * (width * 2 + space) + width * 0.5, regions, rotation=90)
    
    if ymax is not None:
        plt.ylim(ymin, ymax)
    else:
        plt.legend()
    
    sns.despine()
    return plt.ylim()


def generate_plots(args):
    """Generate all visualization plots"""
    print("Generating plots...")
    
    # Check if data exists
    cache_keys = [
        f"cache.tests.paper_plots.f3.perc.{args.clip_duration}",
        f"cache.tests.paper_plots.f3.behaviour.{args.clip_duration}",
    ]
    
    for key in cache_keys:
        if not pickle_store.exists(key):
            print(f"=" * 70)
            print(f"WARNING: {key} not found!")
            print("=" * 70)
            print("Please run with --mode compute first")
            return
    
    # Load data
    roi_perc = pickle_store.load(cache_keys[0])
    beh_vals = pickle_store.load(cache_keys[1])
    
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Get regions
    regions = [r for s in STREAM_CONSIDERED for r in STREAMS[s]]
    stream_indices = [[regions.index(r) for r in STREAMS[s]] for s in STREAM_CONSIDERED]
    
    # ROI bar plots
    print("Plotting ROI bars...")
    plt.figure(figsize=(10, 2))
    img = np.array([roi_perc[r][0] for r in regions])
    afd = np.array([roi_perc[r][1] for r in regions])
    combined = np.array([roi_perc[r][2] for r in regions])
    ylims = plot_bar(img, afd, combined, regions)
    plt.savefig(os.path.join(output_dir, f"roi_bars_{args.clip_duration}.svg"), 
               bbox_inches="tight", transparent=True)
    plt.close()
    
    # Stream bar plots
    print("Plotting stream bars...")
    plt.figure(figsize=(1.1, 2))
    img_streams = np.array([img[s].mean(0) for s in stream_indices])
    afd_streams = np.array([afd[s].mean(0) for s in stream_indices])
    combined_streams = np.array([combined[s].mean(0) for s in stream_indices])
    plot_bar(img_streams, afd_streams, combined_streams, STREAM_CONSIDERED, *ylims)
    plt.savefig(os.path.join(output_dir, f"stream_bars_{args.clip_duration}.svg"), 
               bbox_inches="tight", transparent=True)
    plt.close()
    
    # Behaviour plots
    print("Plotting behaviour bars...")
    conditions = ["normal", "shuffle", "average"]
    img_beh = np.array([beh_vals[r][0] for r in conditions])
    afd_beh = np.array([beh_vals[r][1] for r in conditions])
    combined_beh = np.array([beh_vals[r][2] for r in conditions])
    plt.figure(figsize=(1.1, 2))
    plot_bar(img_beh, afd_beh, combined_beh, conditions, *ylims)
    plt.savefig(os.path.join(output_dir, f"behaviour_bars_{args.clip_duration}.svg"), 
               bbox_inches="tight", transparent=True)
    plt.close()
    
    print(f"All plots saved to {output_dir}/")


def main(args):
    """Main analysis pipeline"""
    if args.mode == "compute":
        compute_roi_voxel_percentages(args)
        compute_behaviour_percentages(args)
    elif args.mode == "plot":
        generate_plots(args)
    elif args.mode == "all":
        compute_roi_voxel_percentages(args)
        compute_behaviour_percentages(args)
        generate_plots(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    extra_args = [
        (("--mode",), {"type": str, "default": "plot",
                       "choices": ["compute", "plot", "all"],
                       "help": "Mode: 'compute' for analysis, 'plot' for visualization, 'all' for both"}),
        (("--num-resample",), {"type": int, "default": 200,
                              "help": "Number of resamples for R² estimation"}),
        (("--topk",), {"type": int, "default": 10,
                      "help": "Top k models for layer mapping"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots"}),
    ]
    
    args = get_args(*extra_args)
    
    # Set tasks to test
    args.tasks_to_test = DEFAULT_TASKS_TO_TEST
    
    main(args)