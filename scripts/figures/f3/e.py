import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.utils import *
from src.store import pickle_store

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f3e')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Task categories for grouped analysis
TASK_CATEGORIES = {
    "All Tasks": ['all'],
    "Static": [
        "imagenet2012",
        "vggface2",
        "majajhong2015-pose",
    ],
    "Hybrid": [
        "mcmahon2023-social",
        "kinetics400",
        "smthsmthv2",
    ],
    "Dynamic": [
        "selfmotion",
        "hdm05",
        "afd2022",
        "ding2012",
    ]
}


def check_data_computed(args):
    """Check if required data has been computed, warn user if not"""
    cache_key = f"cache.tests.paper_plots.f2.{args.clip_duration}"
    try:
        data = pickle_store.load(cache_key)
        return True
    except:
        print("=" * 70)
        print("WARNING: Required data has not been computed yet!")
        print("=" * 70)
        print(f"Please run the computation script first (e.g., 'c.py') with:")
        print(f"  python c.py --clip-duration {args.clip_duration}")
        print("=" * 70)
        return False


def compute_neural_task_relevance(args):
    """Compute task relevance for neural data - wrapper function"""
    print("=" * 70)
    print("NOTE: This script is for plotting only.")
    print("Please run f3.c first to compute the required data.")
    print("=" * 70)
    
    # Check if data exists
    if not check_data_computed(args):
        return None
    
    return pickle_store.load(f"cache.tests.paper_plots.f2.{args.clip_duration}")


def plot_bar_comparison(data_to_plot, args):
    """Create grouped bar comparison plot"""
    groups = args.task_groups or TASK_CATEGORIES
    
    types = [
        "self",
        "imagenet2012",
        "imagenet2012+afd2022",
        "imagenet2012+hdm05",
    ]

    interval = 2
    width = 0.3
    ls = {}
    colors = [MIXED, DYNAMIC, BASELINE_2, BASELINE_1]
    
    for i, group in enumerate(groups.keys()):
        # Get categories for this group
        categories = groups[group]
        
        # Check if all categories have subtraction data
        valid_categories = [c for c in categories if f"sub:{c}" in data_to_plot]
        if not valid_categories:
            continue
            
        r2s = [data_to_plot[f"sub:{cate}"] for cate in valid_categories]
        r2s = list(zip(*r2s))
        r2s = [np.mean(np.array(t)[..., 0], 0) for t in r2s]
        
        for j, (type_name, r2, c) in enumerate(zip(types, r2s, colors)):
            if type_name not in ls:
                ls[type_name] = [[], [], []]
            x = i * interval + j * width
            y = r2[0].item()
            ls[type_name][0].append(x)
            ls[type_name][1].append(y)
            ls[type_name][2].append(c)

            # Annotate significance
            if r2[1] > 0.05:
                plt.text(x - width/2.02, max(y, 0) + 0.03, "n.s.", 
                        fontsize=5, verticalalignment="center")
        
        # Print reduction statistics for first group (All Tasks)
        if i == 0:
            a, b, c, d = [ls[type_name][1][0] for type_name in types]
            print(f"\nReduction statistics:")
            print(f"  By imagenet2012: {(a-b)/a:.4f} ({(a-b)/a*100:.1f}%)")
            print(f"  By imagenet2012+afd2022: {(a-c)/a:.4f} ({(a-c)/a*100:.1f}%)")
            print(f"  By imagenet2012+hdm05: {(a-d)/a:.4f} ({(a-d)/a*100:.1f}%)")

    # Plot bars
    for type_name, (x, y, c) in ls.items():
        plt.bar(x, y, width, label=type_name, color=c)

    plt.xticks([i * interval + (len(types) - 1) * width / 2 for i in range(len(groups))], 
              groups.keys())
    plt.ylabel("Remaining Relevance (R²)")
    plt.legend(fontsize=8)
    sns.despine()
    plt.xticks(rotation=45, ha="right")


def plot_comparison(args):
    """Generate comparison plot with grouped categories"""
    print("Generating grouped comparison plot...")
    
    # Check if data has been computed
    if not check_data_computed(args):
        return
    
    cache_key = f"cache.tests.paper_plots.f2.{args.clip_duration}"
    data = pickle_store.load(cache_key)

    data_to_plot = {k: v for k, v in data.items() if "sub" in k}
    
    if not data_to_plot:
        print("No subtraction data found to plot")
        return

    # Create plot
    plt.figure(figsize=(args.fig_width, args.fig_height))
    plot_bar_comparison(data_to_plot, args)
    plt.xlim(-0.5)
    plt.ylim(0, args.ylim)
    plt.tight_layout()

    # Save plot
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = f"comparison_{args.clip_duration}"
    plt.savefig(os.path.join(output_dir, f"{output_prefix}.svg"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved {output_prefix} plots to {output_dir}")


def main(args):
    if compute_neural_task_relevance(args):
        plot_comparison(args)


if __name__ == "__main__":
    extra_args = [
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots (default: script directory)"}),
        (("--ylim",), {"type": float, "default": 0.8,
                       "help": "Y-axis limit for comparison plot"}),
        (("--fig-width",), {"type": float, "default": 4.2,
                           "help": "Figure width for comparison plot"}),
        (("--fig-height",), {"type": float, "default": 3,
                            "help": "Figure height for comparison plot"}),
    ]
    
    args = get_args(*extra_args)
    
    # Set task groups for comparison
    args.task_groups = TASK_CATEGORIES
    
    main(args)