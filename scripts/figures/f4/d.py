import os
import numpy as np
from matplotlib import pyplot as plt

from scripts.utils import *

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f4d')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Configuration constants
NUM_BEST_MODELS = 4

TASK_NAME_MAPPING = {
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

TASKS_ORDER = [
    "vggface2",
    "imagenet2012",
    "majajhong2015-pose",
    "kinetics400",
    "mcmahon2023-social",
    "smthsmthv2",
    "selfmotion",
    "ding2012",
    "afd2022",
    "hdm05",
]

TASK_COLORS = [
    STATIC, STATIC, STATIC,
    MIXED, MIXED, MIXED,
    DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC,
]

BEH_ORDERING = ["han2024-RGB", "han2024-RGB-S", "han2024-J-6P", "han2024-J-6P-S"]

FMRI_ORDERING = [
    "V1", "V2", "V3",
    "PIT", "FFC", "VVC",
    "V3A", "V6", "VIP",
    "LO1", "MT", "FST",
]


def prepare_task_scores(task, models):
    """Compute task scores for all models across all tasks"""
    return np.array([[task[t][m].mean(0) for m in models] for t in TASKS_ORDER]).T


def prepare_beh_scores(beh, models):
    """Compute behavioral scores for all models"""
    return np.array([[np.nanmean(beh[b][m][0], 0) for m in models] for b in BEH_ORDERING]).T


def prepare_fmri_scores(fmri, models, ceiling):
    """Compute fMRI scores for all models across brain regions"""
    fmri_scores_raw = np.array([fmri[m].mean(0) for m in models])
    
    fmri_scores = []
    for region in FMRI_ORDERING:
        region_scores = select_and_ceil(fmri_scores_raw, ceiling, region).reshape(-1, 1)
        fmri_scores.append(region_scores)
    
    return np.concatenate(fmri_scores, axis=1)


def filter_valid_data(scores, models):
    """Remove models with NaN scores"""
    valid_mask = ~np.isnan(scores).any(axis=1)
    return scores[valid_mask], np.array(models)[valid_mask]


def select_top_models(scores, models, model_of_interests=None):
    """Select either models of interest or top performing models"""
    if model_of_interests is not None:
        indices = np.array([list(models).index(m) for m in model_of_interests])
    else:
        avg_scores = scores.mean(1)
        indices = np.argsort(avg_scores)[-NUM_BEST_MODELS:]
    
    return indices, scores[indices]


def compute_reference_scores(scores):
    """Compute population statistics for reference lines"""
    return {
        'mean': np.mean(scores, axis=0),
        'best': scores.max(0)
    }


def setup_radar_plot():
    """Create and configure radar plot axes"""
    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(4.3, 4.3), subplot_kw=dict(polar=True))
    ax.spines['polar'].set_visible(False)
    return fig, ax


def plot_model_lines(ax, angles, scores, models, indices):
    """Plot lines for selected models"""
    linestyles = ['dotted', 'dotted', 'dotted', 'dotted', 'dotted']
    markers = ['o', 's', '^', 'D', 'x']
    
    for i, idx in enumerate(indices):
        model_name = models[idx]
        values = scores[i].tolist() + scores[i].tolist()[:1]  # close the loop
        
        ax.plot(
            angles, values,
            label=model_name,
            color=get_color_by_model(model_name),
            linewidth=1,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5
        )


def plot_reference_lines(ax, angles, reference_scores):
    """Plot reference lines for best model performance"""
    best_values = reference_scores['best'].tolist() + reference_scores['best'].tolist()[:1]
    ax.plot(
        angles, best_values,
        label="Best Model",
        color=BASELINE_2,
        linewidth=1,
        linestyle='--'
    )


def configure_axes(ax, angles, ordering, scores):
    """Configure axis labels, ticks, and limits"""
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ordering)
    
    # Set radial ticks
    max_score = np.nanmax(scores)
    min_score = 0
    yticks = np.linspace(min_score, max_score, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:.2f}" for tick in yticks])
    ax.set_ylim(min_score, max_score)
    
    # Position legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))


def save_radar_plot(name):
    """Save the radar plot to file"""
    output_path = os.path.join(FIGURE_DIR, f"radar_{name}.svg")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved radar plot: {output_path}")


def make_radar(name, scores, models, ordering, model_of_interests=MODEL_OF_INTEREST):
    """Generate a radar plot for model performance comparison"""
    # Filter and prepare data
    scores, models = filter_valid_data(scores, models)
    indices, selected_scores = select_top_models(scores, models, model_of_interests)
    reference_scores = compute_reference_scores(scores)
    
    # Setup plot geometry
    num_vars = len(ordering)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    
    # Create plot
    fig, ax = setup_radar_plot()
    plot_model_lines(ax, angles, selected_scores, models, indices)
    plot_reference_lines(ax, angles, reference_scores)
    configure_axes(ax, angles, ordering, scores)
    
    # Save
    save_radar_plot(name)


def main(args):
    """Main function to generate radar plots for task, behavior, and fMRI data"""
    # Load data
    task, beh, fmri, models = collect(args, types=['task', 'beh', 'fmri'])
    ceiling = get_ceiling(args)
    
    # Generate task radar plot
    task_scores = prepare_task_scores(task, models)
    make_radar("task", task_scores, models, ordering=TASKS_ORDER)
    
    # Generate behavioral radar plot
    beh_scores = prepare_beh_scores(beh, models)
    make_radar("beh", beh_scores, models, ordering=BEH_ORDERING)
    
    # Generate fMRI radar plot
    fmri_scores = prepare_fmri_scores(fmri, models, ceiling)
    make_radar("fmri", fmri_scores, models, ordering=FMRI_ORDERING)


if __name__ == "__main__":
    args = get_args()
    main(args)