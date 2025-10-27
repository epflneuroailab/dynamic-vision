import os
import numpy as np
from matplotlib import pyplot as plt

from scripts.utils import *
from src.store import pickle_store
from src.analysis.visualize import plot_dual

# Configuration constants
N_PERM = 1000

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f5abc')
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_cached_results(args):
    """Load cached permutation analysis results with validation"""
    cache_key = f"cache.tests.paper_plots.f4.map.{args.clip_duration}.perm{N_PERM}"
    
    try:
        results = pickle_store.load(cache_key)
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(
            f"Cache not found! Please run f4.e first:\n"
            f"  Missing: {cache_key}\n"
            f"Error: {e}"
        )
    
    return results


def extract_r2_scores(results):
    """Extract and clip R² scores from cached results"""
    r2_all, r2_img, r2_afd, unique_img, unique_afd = results
    
    # Extract values and clip to valid range [0, 1]
    r2_img_scores = np.clip(r2_img[0], 0, 1)
    r2_afd_scores = np.clip(r2_afd[0], 0, 1)
    
    return r2_all, r2_img_scores, r2_afd_scores


def compute_motion_index(r2_img, r2_afd):
    """Compute motion index as ratio of action to total R²"""
    return r2_afd / (r2_afd + r2_img + 1e-10)


def apply_ceiling_mask(values, ceiling):
    """Mask out values where ceiling is invalid"""
    masked_values = values.copy()
    masked_values[np.isnan(ceiling)] = np.nan
    return masked_values


def generate_motion_map(args):
    """Generate and save motion index brain map"""
    # Load data
    ceiling = get_ceiling(args)
    results = load_cached_results(args)
    
    # Extract R² scores
    r2_all, r2_img, r2_afd = extract_r2_scores(results)
    
    # Compute motion index
    motion_index = compute_motion_index(r2_img, r2_afd)
    
    # Apply ceiling mask
    motion_index_masked = apply_ceiling_mask(motion_index, ceiling)
    motion_mask = motion_index_masked >= 0.5
    object_mask = motion_index_masked <= 0.5

    print("Please get (a) from f4.f.")
    
    # Generate visualization
    index = motion_index_masked.copy()
    index[~object_mask] = np.nan
    plot_dual(index)
    output_path = os.path.join(FIGURE_DIR, "map_object.png")
    plt.savefig(output_path, dpi=600, transparent=True)
    plt.close()

    print(f"Object-biased map (b) saved to: {output_path}")

    # Generate visualization
    index = motion_index_masked.copy()
    index[~motion_mask] = np.nan
    plot_dual(index)
    output_path = os.path.join(FIGURE_DIR, "map_motion.png")
    plt.savefig(output_path, dpi=600, transparent=True)
    plt.close()
    
    print(f"Motion-biased map (c) saved to: {output_path}")
    print(f"Use pycortex web to visualize the 3D maps.")



def main():
    """Main entry point for motion map generation"""
    args = get_args()
    
    try:
        generate_motion_map(args)
    except RuntimeError as e:
        print(f"\n⚠️  ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()