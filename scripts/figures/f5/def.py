import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from scripts.utils import *
from src.store import pickle_store
from src.analysis.fdr import false_discovery_control
from scipy.stats import pearsonr

plt.rcParams['svg.fonttype'] = 'none'

SCALE = 1.4
FIGSIZE = (3*SCALE, 2*SCALE)

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f5def')
os.makedirs(FIGURE_DIR, exist_ok=True)

F5_ROI_MAP = {
    "Early": ["V1", "V2", "V3"],
    "V3A": ["V3A"],
    "IPS": ["IP0", "IP1", "IP2"],
    "area 7": ["7AL", "7Am", "7PC", "7Pl", "7Pm"],
    "LO": ["LO1", "LO2", "LO3"],
    "MT+": ["MT", "MST"],
    "pSTS": ["STV"],
}

GROUPS = [
    ["Early",],
    ["V3A", "IPS", "area 7"],
    ["LO", "MT+", "pSTS"],
]

AVOID = ["han2024-RGB-4F", "han2024-J-6P-4F"]


def compute_behaviour_map(args):
    """Compute behaviour relevance map from fMRI and behaviour scores"""
    print("Computing behaviour map...")
    
    fmri, beh, models = collect(args, types=['fmri', 'beh'])
    ceiling = get_ceiling(args)
    
    fmri_scores = np.array([fmri[m].mean(0) for m in models])

    # behaviour scores
    beh_scores = {
        beh_name: np.array([np.nanmean(beh[beh_name][m][0], 0) for m in models]) 
        for beh_name in beh if beh_name not in AVOID
    }

    aver_beh_scores = np.array([val for val in beh_scores.values()]).mean(0)

    conditions = []
    for condition in ["shuffle", "normal"]:
        beh_scores_ = []
        for beh_name in beh_scores:
            if condition == "shuffle" and beh_name.endswith("-S"):
                beh_scores_.append(beh_scores[beh_name])
            if condition == "normal" and (beh_name.endswith("RGB") or beh_name.endswith("6P")):
                beh_scores_.append(beh_scores[beh_name])

        beh_score = np.array(beh_scores_).mean(0)
        nans = np.isnan(beh_score)
        beh_score_ = beh_score[~nans]
        fmri_scores_ = fmri_scores[~nans]

        r2, = permutation_analyze(beh_score_.reshape(-1, 1), fmri_scores_, n_permutations=args.num_perm)
        conditions.append(r2)

    nans = np.isnan(aver_beh_scores)
    aver_beh_scores_ = aver_beh_scores[~nans]

    r2, = permutation_analyze(aver_beh_scores_.reshape(-1, 1), fmri_scores[~nans], n_permutations=args.num_perm)

    cache_key = f"cache.tests.paper_plots.f5.map_behaviour.{args.clip_duration}"
    pickle_store.store((r2, conditions), cache_key)
    print(f"Stored behaviour map to {cache_key}")
    
    return r2, conditions


def plot_behaviour_map(args):
    """Plot behaviour map on brain surface"""
    print("Generating behaviour map plots...")
    
    ceiling = get_ceiling(args)
    
    cache_key = f"cache.tests.paper_plots.f5.map_behaviour.{args.clip_duration}"
    
    try:
        map_vals, p = pickle_store.load(cache_key)
        map_vals = map_vals[0] if isinstance(map_vals, tuple) else map_vals
    except Exception as e:
        print(f"Error loading cached data: {e}")
        print(f"Please run with --mode compute first to generate the behaviour map.")
        return
    
    map_vals[np.isnan(ceiling)] = np.nan
    
    output_dir = args.output_dir or FIGURE_DIR
    beh_map_dir = os.path.join(output_dir, "beh_map")
    os.makedirs(beh_map_dir, exist_ok=True)
    
    # Plot using plot_single_factor
    plot_single_factor(map_vals, vmin=0, with_colorbar=True, cmap="Blues")
    plt.savefig(os.path.join(beh_map_dir, "aver.png"), dpi=600, transparent=True)
    plt.close()
    print(f"Saved behaviour map average plot to {beh_map_dir}")
    
    # Plot using plot_volume_nilearn for different views
    for hemi, view in plot_volume_nilearn(map_vals, cmap="Blues"):
        if hemi == 'right' and view == 'lateral':
            plt.savefig(os.path.join(beh_map_dir, f"{hemi}_{view}.png"), 
                       dpi=600, transparent=True, bbox_inches='tight')
            plt.close()
            print(f"Saved {hemi}_{view}.png to {beh_map_dir}")
            break


def _plot(data, data_std, colors):
    """Helper function to plot grouped data with error bars"""
    offsets = []
    offset = 0
    for i, (group, color) in enumerate(zip(GROUPS, colors)):
        group_vals = data[offset:offset + len(group)]
        group_stds = data_std[offset:offset + len(group)]
        plt.plot(np.arange(len(group_vals)) + offset, group_vals, marker='o', color=color)
        plt.errorbar(np.arange(len(group_vals)) + offset, group_vals, yerr=group_stds, 
                    marker='o', capsize=5, color=color)
        offset += len(group_vals)
        offsets.append(offset)
    
    early_vals = data[0]
    for i, (group, offset, color) in enumerate(zip(GROUPS[1:], offsets[:-1], colors[1:])):
        group_start = data[offset]
        plt.plot([0, offset], [early_vals, group_start], linestyle='--', color=color)

    plt.xticks(range(len(F5_ROI_MAP)), list(F5_ROI_MAP.keys()))


def plot_roi_analysis(args):
    """Plot ROI-based analysis of behaviour, layer, motion, and object relevance"""
    print("Generating ROI analysis plots...")
    
    ceiling = get_ceiling(args)
    
    # Load cached data
    beh_cache_key = f"cache.tests.paper_plots.f5.map_behaviour.{args.clip_duration}"
    layer_cache_key = "cache.tests.paper_plots.f1.layer"
    map_cache_key = f"cache.tests.paper_plots.f4.map.{args.clip_duration}.perm1000"
    
    try:
        beh_vals, p = pickle_store.load(beh_cache_key)[0]
        layer = pickle_store.load(layer_cache_key)
        r2_all, r2_img, r2_afd, unique_img, unique_afd = pickle_store.load(map_cache_key)
    except Exception as e:
        print(f"Error loading cached data: {e}")
        print("Please ensure the required cached data (including f2.e) exists.")
        return
    
    r2_all = r2_all[0]
    r2_img = r2_img[0]
    r2_afd = r2_afd[0]

    left_hemi = slice(0, 10242)
    beh_vals[np.isnan(ceiling)] = np.nan
    beh_vals[left_hemi] = np.nan

    # Collect statistics for each ROI
    layer_vals = []
    beh_reles = []
    motion_reles = []
    object_reles = []

    layer_vals_std = []
    beh_reles_std = []
    motion_reles_std = []
    object_reles_std = []

    for name, region in F5_ROI_MAP.items():
        voxels = region_voxels(region)
        beh_reles.append(np.nanmedian(beh_vals[voxels]))
        layer_vals.append(np.nanmedian(layer[voxels]))
        motion_reles.append(np.nanmedian(r2_afd[voxels]))
        object_reles.append(np.nanmedian(r2_img[voxels]))

        beh_reles_std.append(np.nanstd(beh_vals[voxels]))
        layer_vals_std.append(np.nanstd(layer[voxels]))
        motion_reles_std.append(np.nanstd(r2_afd[voxels]))
        object_reles_std.append(np.nanstd(r2_img[voxels]))

        print(f"{name}: \tBeh rele: {np.nanmedian(beh_vals[voxels]):.3f}, "
              f"\tlayer: {np.nanmean(layer[voxels]):.3f}, "
              f"\tmotion rele: {np.nanmedian(r2_afd[voxels]):.3f}, "
              f"\tobject rele: {np.nanmedian(r2_img[voxels]):.3f}")

    layer_vals = np.array(layer_vals)
    beh_reles = np.array(beh_reles)
    motion_reles = np.array(motion_reles)
    object_reles = np.array(object_reles)

    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Plot behaviour relevance
    plt.figure(figsize=FIGSIZE)
    _plot(beh_reles, beh_reles_std, [BASELINE_1, "#99C1F1", "#1D72D8"])
    sns.despine()
    plt.ylabel("Behaviour Relevance")
    plt.savefig(os.path.join(output_dir, "f5_beh.svg"), bbox_inches='tight')
    plt.close()
    print(f"Saved f5_beh plots to {output_dir}")

    # Plot layer values
    plt.figure(figsize=FIGSIZE)
    _plot(layer_vals, layer_vals_std, [BASELINE_1, "#99C1F1", "#1D72D8"])
    sns.despine()
    plt.ylabel("Hierarchy")
    plt.savefig(os.path.join(output_dir, "f5_layer.svg"), bbox_inches='tight')
    plt.close()
    print(f"Saved f5_layer plots to {output_dir}")


def plot_brain_regions(args):
    """Plot brain surface with ROI labels"""
    print("Generating brain surface region plot...")
    
    ceiling = get_ceiling(args)
    
    region_vals = np.zeros(20484) * np.nan

    for i, (name, regions) in enumerate(F5_ROI_MAP.items()):
        voxels = region_voxels(regions)
        region_vals[voxels] = i + 1

    region_vals[np.isnan(ceiling)] = np.nan
    
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    for hemi in plot_surface_nilearn(region_vals, cmap="inferno"):
        if hemi == 'right':
            plt.savefig(os.path.join(output_dir, f"{hemi}_region.png"), 
                       dpi=600, bbox_inches='tight')
            plt.close()
            print(f"Saved {hemi}_region.png to {output_dir}")
            break


def main(args):
    if args.mode == "compute":
        compute_behaviour_map(args)
    elif args.mode == "plot_behaviour_map":
        plot_behaviour_map(args)
    elif args.mode == "plot_roi":
        plot_roi_analysis(args)
    elif args.mode == "plot_brain":
        plot_brain_regions(args)
    elif args.mode == "plot":
        plot_behaviour_map(args)
        plot_roi_analysis(args)
        plot_brain_regions(args)
    elif args.mode == "all":
        compute_behaviour_map(args)
        plot_behaviour_map(args)
        plot_roi_analysis(args)
        plot_brain_regions(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. "
                        "Choose 'compute', 'plot_behaviour_map', 'plot_roi', "
                        "'plot_brain', 'plot', or 'all'")


if __name__ == "__main__":
    extra_args = [
        (("--mode",), {"type": str, "default": "plot", 
                       "choices": ["compute", "plot_behaviour_map", "plot_roi", 
                                   "plot_brain", "plot", "all"],
                       "help": "Mode: 'compute' to compute behaviour map, "
                              "'plot_behaviour_map' for behaviour map visualization, "
                              "'plot_roi' for ROI plots, 'plot_brain' for brain surface plot, "
                              "'plot' for all plots, 'all' for compute + all plots"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots (default: script directory)"}),
    ]
    
    args = get_args(*extra_args)
    main(args)