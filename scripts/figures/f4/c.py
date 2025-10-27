import os
import numpy as np
from matplotlib import pyplot as plt

from scripts.utils import *


FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f4c')
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_and_prepare_data(args):
    """Load data and compute scores for models"""
    ceiling = get_ceiling(args)
    fmri, tasks, models = collect(args, types=['fmri', 'task'])
    
    fmri_scores = np.array([fmri[m].mean(0) for m in models])
    fmri_scores = select_and_ceil(fmri_scores, ceiling, "Whole_Brain").reshape(-1)
    
    return fmri_scores, tasks, models


def compute_task_scores(tasks, models, task_name):
    """Compute scores for a specific task across all models"""
    return np.array([tasks[task_name][m].mean(0) for m in models])


def create_scatter_plot(task_a_scores, task_b_scores, fmri_scores, task_a, task_b, scale=0.75):
    """Create base scatter plot with fMRI scores as colormap"""
    plt.figure(figsize=(6.3 * scale, 5 * scale))
    plt.scatter(task_a_scores, task_b_scores, c=fmri_scores, cmap='inferno', 
                s=100, edgecolor='none')
    plt.colorbar()
    plt.xlabel(task_a)
    plt.ylabel(task_b)


def annotate_models(models, task_a_scores, task_b_scores, task_a, task_b):
    """Add model annotations and markers to the plot"""
    for i, model in enumerate(models):
        # Debug print for extra models
        if model in EXTRA_MODELS:
            print(f"{model}, {task_a}: {task_a_scores[i]:.4f}, "
                  f"{task_b}: {task_b_scores[i]:.4f}")
        
        # Choose marker color based on model type
        color = 'blue' if model in IMAGE_MODELS else 'orange'
        
        # Add marker and label
        plt.scatter(task_a_scores[i], task_b_scores[i], c=color, 
                   marker='x', s=0.5)
        plt.text(task_a_scores[i], task_b_scores[i], model, fontsize=5)


def style_plot():
    """Apply styling to remove top and right spines"""
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_plot(output_dir, task_a, task_b, annotate):
    """Save the plot with appropriate filename"""
    suffix = '_annotated' if annotate else ''
    filename = f"{output_dir}/task_{task_a}_{task_b}{suffix}.svg"
    plt.savefig(filename, transparent=True)
    print(f"Plot saved to: {filename}")


def main(args):
    """Main function to generate task comparison scatter plot"""
    
    # Configuration
    task_a = 'imagenet2012'
    task_b = 'afd2022'
    
    # Load and prepare data
    fmri_scores, tasks, models = load_and_prepare_data(args)
    task_a_scores = compute_task_scores(tasks, models, task_a)
    task_b_scores = compute_task_scores(tasks, models, task_b)
    
    # Create visualization
    create_scatter_plot(task_a_scores, task_b_scores, fmri_scores, task_a, task_b)
    
    # Add annotations if requested
    if args.annotate:
        annotate_models(models, task_a_scores, task_b_scores, task_a, task_b)
    
    # Style and save
    style_plot()
    save_plot(FIGURE_DIR, task_a, task_b, args.annotate)
    plt.close()


if __name__ == "__main__":
    args = get_args(
        ("--annotate", {"action": "store_true"}),
    )
    main(args)