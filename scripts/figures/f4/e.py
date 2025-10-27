import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from diptest import diptest
from scipy.ndimage import gaussian_filter

from scripts.utils import *
from src.store import pickle_store

# Configuration constants
MASK_DIR = "scripts/utils/regions/NSD"
N_PERM = 1000
LAYER_NUM_BINS = 25
INDEX_NUM_BINS = 25
SIGNIFICANCE_THRESHOLD = 0.05
HISTOGRAM_BINS = 100
SMOOTH_SIGMA = (1, 1)  # Gaussian smoothing sigma for heatmap

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f4e')
os.makedirs(FIGURE_DIR, exist_ok=True)


def get_stream_mask():
    """Load and process brain stream masks from fsaverage7 to fsaverage5"""
    lh_path = os.path.join(MASK_DIR, "lh.streams.mgz")
    rh_path = os.path.join(MASK_DIR, "rh.streams.mgz")
    
    import nibabel as nib
    lh_img = nib.load(lh_path).get_fdata().flatten()
    rh_img = nib.load(rh_path).get_fdata().flatten()
    
    # Combine hemispheres and mark background
    label = np.concatenate([lh_img, rh_img])
    label[label == 0] = np.nan
    
    # Convert to fsaverage5
    from neuroparc.atlas import Atlas
    label = Atlas("fsaverage7", label).label_surface("fsaverage5")
    
    return label


def compute_statistics(args):
    """Compute permutation analysis and store results"""
    print("Computing statistics...")
    
    # Load data
    fmri, task, models = collect(args, types=['fmri', 'task'])
    ceiling = get_ceiling(args)
    
    # Prepare scores
    task_benchmarks = list(task.keys())
    fmri_scores = np.array([fmri[m].mean(0) for m in models])
    task_scores = np.array([[task[t][m].mean(0) for m in models] for t in task_benchmarks]).T
    
    # Extract task-specific scores
    tasks = ["imagenet2012", "afd2022"]
    toi_scores = task_scores[:, [task_benchmarks.index(t) for t in tasks]]
    img_scores = task_scores[:, [task_benchmarks.index("imagenet2012")]]
    afd_scores = task_scores[:, [task_benchmarks.index("afd2022")]]
    
    # Perform permutation analysis
    print(f"Running permutation analysis with {N_PERM} permutations...")
    r2_all, r2_afd_unique, r2_img_unique = permutation_analyze(
        toi_scores, fmri_scores, img_scores, afd_scores, n_permutations=N_PERM
    )
    
    r2_img_only = permutation_analyze(img_scores, fmri_scores, n_permutations=N_PERM)
    r2_img = r2_img_only[0]
    
    r2_afd_only = permutation_analyze(afd_scores, fmri_scores, n_permutations=N_PERM)
    r2_afd = r2_afd_only[0]
    
    # Store results
    cache_key = f"cache.tests.paper_plots.f4.map.{args.clip_duration}.perm{N_PERM}"
    ceiling_key = f"cache.tests.paper_plots.f4.ceiling.{args.clip_duration}"
    
    pickle_store.store([r2_all, r2_img, r2_afd, r2_img_unique, r2_afd_unique], cache_key)
    pickle_store.store(ceiling, ceiling_key)
    
    print(f"Results saved to cache: {cache_key}")


def load_cached_data(args):
    """Load cached data with validation"""
    cache_key = f"cache.tests.paper_plots.f4.map.{args.clip_duration}.perm{N_PERM}"
    layer_cache_key = "cache.tests.paper_plots.f1.layer"
    
    try:
        r2_data = pickle_store.load(cache_key)
        layer_data = pickle_store.load(layer_cache_key)
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(
            f"Cache not found! Please run with --mode compute first:\n"
            f"  Missing: {cache_key} or {layer_cache_key}\n"
            f"Error: {e}"
        )
    
    return r2_data, layer_data


def prepare_plot_data(args):
    """Load and prepare all required data for plotting"""
    ceiling = get_ceiling(args)
    r2_data, layer_data = load_cached_data(args)
    mask = get_stream_mask()
    
    r2_all, r2_img, r2_afd, unique_img, unique_afd = r2_data
    
    # Clip R² values to valid range
    r2_img_raw = np.clip(r2_img[0], 0, 1)
    r2_afd_raw = np.clip(r2_afd[0], 0, 1)
    
    return ceiling, r2_all, r2_img_raw, r2_afd_raw, mask, layer_data


def compute_motion_index(r2_img_raw, r2_afd_raw):
    """Compute motion index from image and action R² values"""
    return r2_afd_raw / (r2_img_raw + r2_afd_raw + 1e-6)


def filter_valid_voxels(ceiling, mask, r2_all):
    """Identify valid voxels based on ceiling, mask, and significance"""
    valid = (~np.isnan(ceiling)) & (~np.isnan(mask)) & (r2_all[1] < SIGNIFICANCE_THRESHOLD)
    valid_indices = np.where(valid)[0]
    print(f"Number of valid voxels: {np.sum(valid)}")
    return valid, valid_indices


def create_heatmap_data(layer, motion_index):
    """Create 2D histogram (heatmap) of layer vs motion index"""
    heatmap = np.zeros((LAYER_NUM_BINS, INDEX_NUM_BINS))
    
    layer_bins = np.linspace(np.nanmin(layer), np.nanmax(layer), LAYER_NUM_BINS + 1)
    index_bins = np.linspace(0, 1, INDEX_NUM_BINS + 1)
    
    for i in range(LAYER_NUM_BINS):
        for j in range(INDEX_NUM_BINS):
            mask = (
                (layer >= layer_bins[i]) & (layer <= layer_bins[i + 1]) &
                (motion_index >= index_bins[j]) & (motion_index <= index_bins[j + 1])
            )
            heatmap[i, j] = np.sum(mask)
    
    return heatmap, layer_bins, index_bins


def test_unimodality(data, label="Data"):
    """Perform diptest for unimodality"""
    dip, p_value = diptest(data)
    
    if p_value < SIGNIFICANCE_THRESHOLD:
        print(f"{label}: Reject null hypothesis - Evidence against unimodality.")
    else:
        print(f"{label}: Fail to reject null hypothesis - Data is consistent with unimodality.")
    
    print(f"{label} dip test p-value: {p_value:.3f}")
    return dip, p_value


def plot_distribution(motion_index, output_dir):
    """Plot overall distribution of motion index with KDE"""
    scale = 1
    plt.figure(figsize=(4 * scale, 2.5 * scale))
    
    sns.histplot(
        motion_index,
        bins=HISTOGRAM_BINS,
        kde=True,
        color="black",
        stat="count",
        edgecolor=None,
        linewidth=0
    )
    
    # Test for unimodality
    test_unimodality(motion_index, "Overall motion index")
    
    plt.xlabel("Motion index")
    plt.ylabel("Voxel count")
    plt.ylim(0, 100)
    plt.xlim(0, 1)
    sns.despine()
    
    output_path = os.path.join(output_dir, "stats3_hist.svg")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def normalize_heatmap(heatmap):
    """Normalize heatmap by layer bin maximum"""
    return heatmap / np.max(heatmap, axis=1, keepdims=True)


def smooth_heatmap(heatmap, sigma=SMOOTH_SIGMA):
    """Apply Gaussian smoothing to heatmap"""
    return gaussian_filter(heatmap, sigma=sigma)


def plot_heatmap(heatmap, layer_bins, index_bins, output_dir, smooth=False):
    """Plot normalized heatmap of layer vs motion index
    
    Args:
        heatmap: Raw heatmap data
        layer_bins: Layer bin edges
        index_bins: Motion index bin edges
        output_dir: Directory to save output
        smooth: Whether to apply Gaussian smoothing
    """
    # Apply smoothing if requested
    if smooth:
        heatmap = smooth_heatmap(heatmap)
    
    # Normalize
    heatmap_normalized = normalize_heatmap(heatmap)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_normalized.T)
    
    plt.ylabel("Motion index")
    plt.xlabel("Model-mapped hierarchy")
    
    # Configure ticks
    plt.xticks(
        ticks=np.arange(LAYER_NUM_BINS) + 0.5,
        labels=[f"{layer_bins[i]:.2f}" for i in range(LAYER_NUM_BINS)],
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(INDEX_NUM_BINS) + 0.5,
        labels=[f"{index_bins[i]:.2f}" for i in range(INDEX_NUM_BINS)],
        rotation=0
    )
    plt.gca().invert_yaxis()
    
    # Save with appropriate filename
    suffix = "_smooth" if smooth else "_raw"
    output_path = os.path.join(output_dir, f"stats3_heatmap{suffix}.svg")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_layer_distributions(heatmap, layer_bins, index_bins, output_dir):
    """Create individual distribution plots for each layer bin"""
    target_folder = os.path.join(output_dir, "stats3_layers")
    os.makedirs(target_folder, exist_ok=True)
    
    for layer_idx in range(LAYER_NUM_BINS):
        layer_distribution = heatmap[layer_idx, :]
        _, p_value = diptest(layer_distribution)
        
        plt.figure(figsize=(6, 4))
        plt.plot(index_bins[:-1], layer_distribution, color="black", lw=2)
        
        plt.xlabel("Motion index")
        plt.ylabel("Density")
        plt.title(
            f"Layer ({layer_bins[layer_idx]:.2f})\n"
            f"Dip test p-value: {p_value:.3f}"
        )
        
        # Mark significant bimodality
        if p_value < SIGNIFICANCE_THRESHOLD:
            plt.text(
                0.5, 0.9, "*",
                transform=plt.gca().transAxes,
                color="red",
                fontsize=12,
                ha='center',
                va='center'
            )
        
        output_path = os.path.join(target_folder, f"layer_{layer_idx + 1}.svg")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {LAYER_NUM_BINS} layer distributions to: {target_folder}")


def generate_plots(args):
    """Main function to generate motion index analysis plots"""
    print("Generating plots...")
    
    # Load and prepare data
    ceiling, r2_all, r2_img_raw, r2_afd_raw, mask, layer_data = prepare_plot_data(args)
    
    # Compute motion index
    motion_index_full = compute_motion_index(r2_img_raw, r2_afd_raw)
    
    # Filter valid voxels
    valid, valid_indices = filter_valid_voxels(ceiling, mask, r2_all)
    motion_index = motion_index_full[valid]
    layer = layer_data[valid]
    
    # Generate plots
    plot_distribution(motion_index, FIGURE_DIR)
    
    # Create heatmap data
    heatmap, layer_bins, index_bins = create_heatmap_data(layer, motion_index)
    
    # Plot both raw and smoothed versions
    plot_heatmap(heatmap, layer_bins, index_bins, FIGURE_DIR, smooth=False)
    plot_heatmap(heatmap, layer_bins, index_bins, FIGURE_DIR, smooth=True)
    
    # Plot per-layer distributions
    plot_per_layer_distributions(heatmap, layer_bins, index_bins, FIGURE_DIR)
    
    print("All plots generated successfully!")


def main(args):
    """Main entry point - routes to compute or plot based on mode"""
    if args.mode == 'compute':
        compute_statistics(args)
    elif args.mode == 'plot':
        generate_plots(args)
    elif args.mode == 'both':
        compute_statistics(args)
        generate_plots(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'compute', 'plot', or 'both'")


if __name__ == "__main__":
    args = get_args(
        ("--mode", {
            "type": str,
            "default": "plot",
            "choices": ["compute", "plot", "both"],
            "help": "Mode: 'compute' to run analysis, 'plot' to generate figures, 'both' for both"
        }),
    )
    
    try:
        main(args)
    except RuntimeError as e:
        print(f"\n⚠️  ERROR: {e}")
        exit(1)