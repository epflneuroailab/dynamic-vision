import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV

from scripts.utils import *
from src.models.groups import *

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f3f')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Model class definitions
MODEL_CLASS_MAP = {
    "Image Recognition": IMAGE_MODELS,
    "Action Recognition": ACTION_RECOGNITION_MODELS,
    "Masked Autoencoder": MASKED_AUTOENCODER_MODELS,
    "Audio-Video": AUDIO_VIDEO_MODELS,
    "Forward Prediction": FORWARD_PREDICTION_MODELS,
    "Text-Video": TEXT_VIDEO_MODELS,
    "Untrained Models": RANDOM_MODELS,
}

# Ordered groups for plotting
PLOT_GROUPS = [
    "Other",
    "Untrained Models",
    "Image Recognition",
    "Forward Prediction",
    "Audio-Video",
    "Action Recognition",
    "Masked Autoencoder",
    "Text-Video",
]

# Colors for each group
GROUP_COLORS = [BASELINE_1, BASELINE_2, *MODELS]

# Default tasks
DEFAULT_TASKS = [
    "imagenet2012",
    "afd2022",
    "kinetics400",
    "ding2012",
    "smthsmthv2",
    "vggface2",
    "hdm05",
    "selfmotion",
    "mcmahon2023-social",
    "majajhong2015-pose",
]

# Task groups for analysis
DEFAULT_TASK_GROUPS = [
    ["imagenet2012"],
    ["afd2022", "imagenet2012"],
    ["afd2022"],
    ["kinetics400"],
    DEFAULT_TASKS,
]


def map_model_class(model):
    """Map a model to its class category"""
    for class_name, model_list in MODEL_CLASS_MAP.items():
        if model in model_list:
            return class_name
    return "Other"


def get_neural_scores(data, ceiling, region, data_type):
    """Extract and process neural scores"""
    models = list(data.keys())
    
    if data_type == 'fmri':
        data_scores = np.array([data[m].mean(0) for m in models])
        data_scores = select_and_ceil(data_scores, ceiling, region).reshape(-1, 1)
    elif data_type == 'elec':
        data_scores = np.array([data[m].mean(0) for m in models])
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    return data_scores


def compute_task_predictions(task_scores, data_scores, task_names):
    """Compute predictions from task scores using Ridge regression"""
    if len(task_names) == 1:
        # Single task: direct correlation
        return task_scores.reshape(-1)
    else:
        # Multiple tasks: fit Ridge regression
        regressor = RidgeCV(alphas=np.logspace(-3, 3, 10))
        regressor.fit(task_scores, data_scores)
        predictions = regressor.predict(task_scores)
        return predictions.flatten(), regressor
    
def compute_adjusted_r2(r2, n_samples, n_features):
    """Compute adjusted R² to account for number of predictors"""
    if n_samples <= n_features + 1:
        return r2
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def plot_model_scatter(xs, ys, model_classes, task_names, args, 
                       test_xs=None, test_ys=None):
    """Create scatter plot with model classes color-coded"""
    # Compute and print statistics
    r2 = np.corrcoef(xs, ys)[0, 1] ** 2
    adj_r2 = compute_adjusted_r2(r2, len(xs), len(task_names))
    print(f"Adjusted R²: {adj_r2:.3f} - Tasks: {', '.join(task_names)}")
    
    # Plot each model class with its color
    marker_size = 18 if args.small_graph else 45
    
    for group, color in zip(PLOT_GROUPS, GROUP_COLORS):
        idx = np.where(np.array(model_classes) == group)[0]
        if len(idx) > 0:
            plt.scatter(xs[idx], ys[idx], label=group, s=marker_size, 
                       edgecolor='none', c=color)
    
    # Plot test data if provided
    if test_xs is not None and test_ys is not None:
        plt.scatter(test_xs, test_ys, marker='x', s=marker_size * 1.5, 
                   c='black', linewidth=2, label='Test Models')
    
    # Labels and styling
    plt.ylabel("Whole Brain fMRI Alignment" if args.type == 'fmri' else "Neural Alignment")
    plt.xlabel("Task Performance")
    
    if not args.small_graph:
        plt.legend(fontsize=8, frameon=False)
    
    sns.despine()


def analyze_task_group(data, task, task_names, ceiling, args, 
                       test_data=None, test_task=None):
    """Analyze a single task group"""
    models = list(data.keys())
    model_classes = [map_model_class(m) for m in models]
    
    # Get task scores
    task_scores = np.array([[task[task_name][m].mean(0) for m in models] 
                           for task_name in task_names]).T
    
    # Get neural scores
    data_scores = get_neural_scores(data, ceiling, args.region, args.type)
    
    # Compute predictions
    result = compute_task_predictions(task_scores, data_scores, task_names)
    if isinstance(result, tuple):
        xs, regressor = result
    else:
        xs = result
        regressor = None
    
    ys = data_scores.flatten()
    
    # Process test data if provided
    test_xs = None
    test_ys = None
    
    if test_data is not None and test_task is not None:
        test_models = list(test_data.keys())
        test_task_scores = np.array([[test_task[task_name][m].mean(0) 
                                     for m in test_models] 
                                    for task_name in task_names]).T
        test_data_scores = get_neural_scores(test_data, ceiling, args.region, args.type)
        
        if len(task_names) == 1:
            test_xs = test_task_scores.flatten()
        else:
            test_xs = regressor.predict(test_task_scores).flatten()
        
        test_ys = test_data_scores.flatten()
    
    # Create plot
    fig_size = (1.3, 1.3) if args.small_graph else (3.8, 3.8)
    plt.figure(figsize=fig_size)
    
    if args.small_graph:
        plt.xticks([0.25, 0.75])
    
    plot_model_scatter(xs, ys, model_classes, task_names, args, test_xs, test_ys)
    
    return xs, ys


def generate_plots(args):
    """Generate scatter plots for different task groups"""
    print("Generating model class scatter plots...")
    
    # Prepare models
    if args.include_extra:
        extra_models = EXTRA_MODELS + FINETUNE_MODELS
        all_models = ALL_MODELS + extra_models
    else:
        all_models = ALL_MODELS
        extra_models = []
    
    # Load data
    ceiling = get_ceiling(args)
    data, task, models = collect(args, types=[args.type, 'task'], models=all_models)
    
    # Prepare task groups
    task_groups = args.task_groups if args.task_groups else DEFAULT_TASK_GROUPS
    
    # Create output directory
    target_dir = os.path.join(
        args.output_dir or FIGURE_DIR,
        f"{args.type}_{args.region}_{args.clip_duration}"
    )
    os.makedirs(target_dir, exist_ok=True)
    
    # Process each task group
    for task_group in task_groups:
        # Prepare training and test data
        if args.type == 'fmri':
            train_data = {m: data[m] for m in models if m in ALL_MODELS}
            test_data = ({m: data[m] for m in models if m in extra_models} 
                        if args.plot_test and extra_models else None)
        elif args.type == 'elec':
            train_data = {m: data[args.region][m] for m in models if m in ALL_MODELS}
            test_data = ({m: data[args.region][m] for m in models if m in extra_models}
                        if args.plot_test and extra_models else None)
        
        # Prepare test task data
        test_task = None
        if test_data is not None:
            test_task = {
                task_name: {m: task[task_name][m] for m in models if m in extra_models}
                for task_name in task_group
            }
        
        # Analyze and plot
        xs, ys = analyze_task_group(
            train_data, task, task_group, ceiling, args,
            test_data, test_task
        )
        
        # Save plot
        task_str = '+'.join(task_group)
        suffix = '_inset' if args.small_graph else ''
        output_file = f"{task_str}_{args.region}{suffix}.svg"
        plt.savefig(os.path.join(target_dir, output_file), bbox_inches="tight")
        plt.close()
        
        print(f"Saved: {output_file}")


def main(args):
    generate_plots(args)
    print(f"\nAll plots saved to: {args.type}_{args.region}_{args.clip_duration}/")


if __name__ == "__main__":
    extra_args = [
        (("--type",), {"type": str, "default": "fmri",
                       "choices": ["fmri", "elec"],
                       "help": "Type of neural data to analyze"}),
        (("--region",), {"type": str, "default": "Whole_Brain",
                        "help": "Brain region to analyze"}),
        (("--plot-test",), {"action": "store_true",
                           "help": "Plot test models (EXTRA_MODELS and FINETUNE_MODELS) with 'x' markers"}),
        (("--small-graph",), {"action": "store_true",
                             "help": "Generate small inset-style graphs"}),
        (("--include-extra",), {"action": "store_true",
                               "help": "Include extra and finetune models in analysis"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots (default: script directory)"}),
        (("--task-groups",), {"type": str, "nargs": "+", "action": "append",
                             "help": "Custom task groups (can be specified multiple times)"}),
    ]
    
    args = get_args(*extra_args)
    main(args)