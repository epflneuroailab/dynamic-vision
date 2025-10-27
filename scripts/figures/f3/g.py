import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from scripts.utils import *
from src.store import pickle_store

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f3g')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Task name mappings for display
TASK_DISPLAY_NAMES = {
    "imagenet2012": "imagenet",
    "afd2022": "afd",
    "kinetics400": "kinetics400",
    "ding2012": "ding2012",
    "smthsmthv2": "smthsmthv2",
    "vggface2": "vggfacev2",
    "hdm05": "hdm05",
    "selfmotion": "airsim",
    "mcmahon2023-social": "mcmahon2023",
    "majajhong2015-pose": "majajhong2015",
}

# ROIs to analyze
DEFAULT_ROIS = [
    "V1",
    "FFC",
    "V3A",
    "MT",
]

# Colors for ROIs
DEFAULT_ROI_COLORS = [
    "#77767B",
    "#613583",
    "#61A1EA",
    "#1D72D8",
]


def cross_correlation(X, Y):
    """Compute normalized cross-correlation between X and Y"""
    # Center the data
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    # Normalize
    X = X / np.linalg.norm(X, axis=0)
    Y = Y / np.linalg.norm(Y, axis=0)
    
    return np.dot(X.T, Y)


def load_or_compute_data(args):
    """Load cached data or compute from scratch"""
    cache_key = "cache.tests.paper_plots.f2.eigen_task"
    
    if pickle_store.exists(cache_key) and not args.recompute:
        print(f"Loading cached data from {cache_key}")
        fmri, task, models, _ = pickle_store.load(cache_key)
        # Recompute ceiling with current args
        ceiling = get_ceiling(args)
    else:
        print("Computing data from scratch...")
        fmri, task, models = collect(args, types=['fmri', 'task'])
        ceiling = get_ceiling(args)
        pickle_store.store((fmri, task, models, ceiling), cache_key)
        print(f"Data cached to {cache_key}")
    
    return fmri, task, models, ceiling


def prepare_data(fmri, task, models, ceiling, tasks):
    """Prepare and normalize fMRI and task data"""
    # Extract fMRI scores
    fmri_scores = np.array([fmri[m].mean(0) for m in models])
    fmri_scores = select_and_ceil(fmri_scores, ceiling)
    
    # Remove NaN voxels
    valid_mask = ~np.isnan(ceiling)
    fmri_scores = fmri_scores[:, valid_mask]
    
    # Extract task scores
    task_scores = np.array([[task[t][m].mean(0) for m in models] 
                           for t in tasks]).T
    
    # Normalize both datasets
    task_scores = (task_scores - task_scores.mean(0)) / task_scores.std(0)
    fmri_scores = (fmri_scores - fmri_scores.mean(0)) / fmri_scores.std(0)
    
    return fmri_scores, task_scores, valid_mask


def create_roi_masks(rois, ceiling):
    """Create boolean masks for each ROI"""
    valid_mask = ~np.isnan(ceiling)
    
    # Create masks for specified ROIs
    roi_masks = []
    for roi in rois:
        full_mask = region_voxels(roi)
        roi_mask = full_mask[valid_mask]
        roi_masks.append(roi_mask)
    
    # Create mask for non-ROI voxels
    all_roi_mask = np.logical_or.reduce(roi_masks)
    non_roi_mask = ~all_roi_mask
    
    return roi_masks, non_roi_mask


def compute_pca_embedding(task_scores, fmri_scores, n_components=2):
    """Compute PCA embedding of voxel representations"""
    # Compute cross-correlation (voxel representation in task space)
    voxel_repr = cross_correlation(task_scores, fmri_scores).T
    
    # Rectify (only positive correlations)
    voxel_repr[voxel_repr < 0] = 0
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    voxel_embed = pca.fit_transform(voxel_repr)
    
    # Shift to remove origin offset
    origin_shift = pca.transform(np.zeros((1, voxel_repr.shape[1])))
    voxel_embed = voxel_embed - origin_shift
    
    # Get principal components (loadings) and explained variance
    pcs = pca.components_.T
    evs = pca.explained_variance_ratio_
    
    return voxel_embed, pcs, evs


def plot_voxel_embedding(voxel_embed, roi_masks, non_roi_mask, rois, roi_colors, args):
    """Plot voxel embeddings colored by ROI"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot non-ROI voxels in gray
    plt.scatter(voxel_embed[non_roi_mask, 0], voxel_embed[non_roi_mask, 1], 
               s=5, alpha=0.15, c='gray', edgecolors='none', label='Other')
    
    # Plot each ROI with its color
    for roi, roi_mask, roi_color in zip(rois, roi_masks, roi_colors):
        plt.scatter(voxel_embed[roi_mask, 0], voxel_embed[roi_mask, 1], 
                   s=8, alpha=1, label=roi, edgecolors='none', c=roi_color)
    
    # Mark origin
    plt.scatter(0, 0, c='r', s=10, marker='x', linewidths=2)
    
    plt.axis('equal')
    plt.axis('off')
    
    if args.show_legend:
        plt.legend(frameon=False, loc='upper right')
    
    # Save plot
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "eigen-task-voxels.png")
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Saved voxel embedding plot to {output_file}")


def plot_task_vectors(pcs, tasks, evs, args):
    """Plot task vectors in PCA space"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    task_display_names = [TASK_DISPLAY_NAMES.get(t, t) for t in tasks]
    
    # Plot task vectors
    for i, (row, task_name) in enumerate(zip(pcs, task_display_names)):
        # Scale vectors for visibility
        row = row * args.vector_scale
        
        # Draw arrow
        plt.arrow(0, 0, row[0], row[1], head_width=0.05, head_length=0.05, 
                 fc='steelblue', ec='steelblue', alpha=0.7, linewidth=1.5)
        
        # Add label
        plt.text(row[0] * 1.06, row[1] * 1.06, task_name, 
                fontsize=10, ha='center', va='center')
    
    # Mark origin
    plt.scatter(0, 0, c='r', s=20, marker='x', linewidths=2, zorder=10)
    
    # Labels
    plt.xlabel(f"PC 1 ({evs[0]:.1%})")
    plt.ylabel(f"PC 2 ({evs[1]:.1%})")
    plt.title("Task Representation PCA")
    
    plt.axis('equal')
    sns.despine()
    
    # Save plot
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "eigen-task-vectors.svg")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved task vector plot to {output_file}")
    print(f"Explained variance: PC1={evs[0]:.1%}, PC2={evs[1]:.1%}")


def print_statistics(pcs, evs, tasks):
    """Print analysis statistics"""
    print("\n" + "=" * 70)
    print("PCA Analysis Results")
    print("=" * 70)
    
    print(f"\nExplained Variance Ratio:")
    for i, ev in enumerate(evs):
        print(f"  PC{i+1}: {ev:.1%}")
    
    print(f"\nTask Loadings on PC1 and PC2:")
    task_names = [TASK_DISPLAY_NAMES.get(t, t) for t in tasks]
    for task_name, loading in zip(task_names, pcs):
        print(f"  {task_name:20s}: PC1={loading[0]:7.3f}, PC2={loading[1]:7.3f}")
    
    print("=" * 70 + "\n")


def main(args):
    """Main analysis pipeline"""
    print("Starting eigen-task analysis...")
    
    # Mode: compute only
    if args.mode == "compute":
        print("Computing and caching data only (no plots)...")
        fmri, task, models, ceiling = load_or_compute_data(args)
        print(f"Data loaded/computed: {len(models)} models")
        print("Done. Run with --mode plot or --mode all to generate visualizations.")
        return
    
    # Mode: plot only
    if args.mode == "plot":
        # Check if cached data exists
        cache_key = "cache.tests.paper_plots.f2.eigen_task"
        if not pickle_store.exists(cache_key):
            print("=" * 70)
            print("ERROR: No cached data found!")
            print("=" * 70)
            print("Please run with --mode compute or --mode all first to generate data.")
            print("=" * 70)
            return
        
        print("Loading cached data for plotting...")
        fmri, task, models, _ = pickle_store.load(cache_key)
        ceiling = get_ceiling(args)
    
    # Mode: all (compute and plot)
    else:  # args.mode == "all"
        fmri, task, models, ceiling = load_or_compute_data(args)
    
    # Get tasks
    tasks = list(TASK_DISPLAY_NAMES.keys())
    
    # Prepare data
    fmri_scores, task_scores, valid_mask = prepare_data(
        fmri, task, models, ceiling, tasks
    )
    
    print(f"Analyzing {len(models)} models, {fmri_scores.shape[1]} voxels, {len(tasks)} tasks")
    
    # Create ROI masks
    rois = args.rois or DEFAULT_ROIS
    roi_colors = args.roi_colors or DEFAULT_ROI_COLORS[:len(rois)]
    roi_masks, non_roi_mask = create_roi_masks(rois, ceiling)
    
    # Compute PCA embedding
    voxel_embed, pcs, evs = compute_pca_embedding(
        task_scores, fmri_scores, n_components=args.n_components
    )
    
    # Print statistics
    print_statistics(pcs, evs, tasks)
    
    # Generate plots (always for 'plot' and 'all' modes)
    plot_voxel_embedding(voxel_embed, roi_masks, non_roi_mask, 
                        rois, roi_colors, args)
    plot_task_vectors(pcs, tasks, evs, args)
    
    # Store results if requested
    if args.store_results:
        results = {
            'voxel_embed': voxel_embed,
            'pcs': pcs,
            'evs': evs,
            'roi_masks': roi_masks,
            'non_roi_mask': non_roi_mask,
        }
        result_key = f"cache.tests.paper_plots.f2.eigen_task_results"
        pickle_store.store(results, result_key)
        print(f"Stored results to {result_key}")


if __name__ == "__main__":
    extra_args = [
        (("--mode",), {"type": str, "default": "plot",
                       "choices": ["compute", "plot", "all"],
                       "help": "Mode: 'compute' for data processing only, 'plot' for plotting only, "
                              "'all' for both"}),
        (("--recompute",), {"action": "store_true",
                           "help": "Force recomputation of cached data"}),
        (("--n-components",), {"type": int, "default": 2,
                              "help": "Number of PCA components"}),
        (("--vector-scale",), {"type": float, "default": 1.3,
                              "help": "Scaling factor for task vectors in plot"}),
        (("--show-legend",), {"action": "store_true",
                             "help": "Show legend in voxel embedding plot"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots"}),
        (("--rois",), {"type": str, "nargs": "+", "default": None,
                      "help": "ROIs to highlight (default: V1, FFC, V3A, MT)"}),
        (("--roi-colors",), {"type": str, "nargs": "+", "default": None,
                            "help": "Colors for ROIs (hex format)"}),
        (("--store-results",), {"action": "store_true",
                               "help": "Store PCA results to cache"}),
    ]
    
    args = get_args(*extra_args)
    main(args)