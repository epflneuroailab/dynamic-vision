import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.utils import *
from src.store import pickle_store

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f3d')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Default tasks to analyze
DEFAULT_TASKS = [
    "imagenet2012",
    "afd2022",
    "kinetics400",
    "kinetics400-static",
    "ding2012",
    "smthsmthv2",
    "vggface2",
    "hdm05",
    "selfmotion",
    "mcmahon2023-social",
    "majajhong2015-pose",
]

# Benchmarks to avoid
REDUNDANT = ["han2024-RGB-4F", "han2024-J-6P-4F"]

# Predefined ordering for plotting
DEFAULT_ORDERING = [
    "ding2012",
    "selfmotion",
    "hdm05",
    "afd2022",
    "vggface2",
    "imagenet2012",
    "smthsmthv2",
    "majajhong2015-pose",
    "mcmahon2023-social",
    "kinetics400",
    "imagenet2012 + hdm05",
    "imagenet2012 + afd2022",
    "kinetics400-static + hdm05",
    "kinetics400-static + afd2022",
    "shuffle-object",
    "shuffle-motion",
    "normal-object",
    "normal-motion",
]


def compute_behaviour_task_relevance(args):
    """Compute task relevance for behaviour alignment using regression analysis"""
    print("Computing behaviour task relevance statistics...")
    
    beh, task, meta, models = collect(args, types=['beh', 'task', 'meta'], exclude_pixels=True)

    # Compute behaviour scores
    beh_scores = {
        beh_name: np.array([np.nanmean(beh[beh_name][m][0], 0) for m in models]) 
        for beh_name in beh if beh_name not in REDUNDANT
    }

    # Validate models (remove those with NaN values)
    tmp = []
    valid_indices = []
    for i in range(len(models)):
        invalid = False
        for benchmark in beh_scores:
            if np.isnan(beh_scores[benchmark][i]):
                invalid = True
                break
        if not invalid:
            tmp.append(models[i])
            valid_indices.append(i)
    
    discarded = [m for i, m in enumerate(models) if i not in valid_indices]
    models = tmp
    
    if discarded:
        print(f"Discarded {len(discarded)} models due to NaN values: {', '.join(discarded)}")

    # Prepare different behaviour score aggregations
    data_scores = np.array([beh_scores[k] for k in beh_scores.keys()]).mean(0)[valid_indices]
    normal_scores = np.array([beh_scores[k] for k in ['han2024-RGB', 'han2024-J-6P']]).mean(0)[valid_indices]
    shuffle_scores = np.array([beh_scores[k] for k in ['han2024-RGB-S', 'han2024-J-6P-S']]).mean(0)[valid_indices]
    
    task_scores = np.array([[task[t][m].mean(0) for m in models] for t in args.tasks]).T
    meta_values = np.array([list(meta[m].values()) for m in models])

    results = {}

    def _get_stat_r2(name, input_scores, output_scores):
        """Compute R² with confidence intervals using resampling"""
        _, r2 = resample_analyze(input_scores, output_scores, n_resamples=args.num_resample)
        mean, ci_l, ci_u = r2
        print(f"{name}: {mean.item():.4f} ({ci_l.item():.4f}, {ci_u.item():.4f})")
        return r2
    
    # Shuffle condition: object and motion
    print("\nShuffle condition:")
    if 'imagenet2012' in args.tasks:
        task_ids = args.tasks.index('imagenet2012')
        results['shuffle-object'] = _get_stat_r2('shuffle-object', 
                                                  task_scores[:, task_ids].reshape(-1, 1), 
                                                  shuffle_scores)
    
    if 'afd2022' in args.tasks:
        task_ids = args.tasks.index('afd2022')
        results['shuffle-motion'] = _get_stat_r2('shuffle-motion', 
                                                 task_scores[:, task_ids].reshape(-1, 1), 
                                                 shuffle_scores)

    # Normal condition: object and motion
    print("\nNormal condition:")
    if 'imagenet2012' in args.tasks:
        task_ids = args.tasks.index('imagenet2012')
        results['normal-object'] = _get_stat_r2('normal-object', 
                                                task_scores[:, task_ids].reshape(-1, 1), 
                                                normal_scores)
    
    if 'afd2022' in args.tasks:
        task_ids = args.tasks.index('afd2022')
        results['normal-motion'] = _get_stat_r2('normal-motion', 
                                               task_scores[:, task_ids].reshape(-1, 1), 
                                               normal_scores)

    # Meta features
    print("\nMeta features:")
    for i, meta_name in enumerate(['fps', 'model size']):
        results[meta_name] = _get_stat_r2(meta_name, 
                                          meta_values[:, i].reshape(-1, 1), 
                                          data_scores)
    
    # Individual tasks
    print("\nIndividual tasks:")
    for i, task_name in enumerate(args.tasks):
        results[task_name] = _get_stat_r2(task_name, 
                                          task_scores[:, i].reshape(-1, 1), 
                                          data_scores)

    # Task combinations
    print("\nTask combinations:")
    task_combinations = [
        ('imagenet2012 + afd2022', ['imagenet2012', 'afd2022']),
        ('kinetics400-static + afd2022', ['kinetics400-static', 'afd2022']),
        ('imagenet2012 + hdm05', ['imagenet2012', 'hdm05']),
        ('kinetics400-static + hdm05', ['kinetics400-static', 'hdm05']),
        ('imagenet2012 + ding2012', ['imagenet2012', 'ding2012']),
    ]
    
    for combo_name, combo_tasks in task_combinations:
        if all(t in args.tasks for t in combo_tasks):
            task_ids = [args.tasks.index(t) for t in combo_tasks]
            results[combo_name] = _get_stat_r2(combo_name, 
                                              task_scores[:, task_ids], 
                                              data_scores)

    # Store results
    cache_key = f"cache.tests.paper_plots.f2.{args.clip_duration}.beh"
    pickle_store.store(results, cache_key)
    print(f"\nStored results to {cache_key}")
    
    return results


def plot_relevance(d, **kwargs):
    """Plot vertical bar chart with error bars"""
    names = list(d.keys())
    means = [v[0] for v in d.values()]
    cis = [(v[1], v[2]) for v in d.values()]
    
    # Vertical bar plot
    plt.bar(names, means, **kwargs)

    # Print statistics
    for name, mean in zip(names, means):
        print(f"{name}: {mean:.3f} ({cis[names.index(name)][0]:.3f}, {cis[names.index(name)][1]:.3f})")

    # Add error bars
    for i, (ci_low, ci_high) in enumerate(cis):
        ci_low = max(ci_low, 0)
        plt.plot([i, i], [ci_low, ci_high], color="black")

    plt.ylim(0)
    sns.despine()


def plot_behaviour_task_relevance(args):
    """Generate behaviour task relevance ranking plot"""
    print("Generating behaviour task relevance plot...")
    
    # Load cached data
    cache_key = f"cache.tests.paper_plots.f2.{args.clip_duration}.beh"
    try:
        data = pickle_store.load(cache_key)
    except Exception as e:
        print(f"Error loading cached data: {e}")
        print("Please run with --mode compute first")
        return

    # Use predefined ordering or sort by values
    if args.use_custom_order:
        ordering = [k for k in DEFAULT_ORDERING if k in data]
        tmp = {k: data[k] for k in ordering + ["fps", "model size"] if k in data}
    else:
        tmp = sorted(data.items(), key=lambda item: item[1][0].item(), reverse=False)
        tmp = {k: v for k, v in tmp}
    
    tmp = {k: [a.item() for a in v] for k, v in tmp.items()}
    
    # Organize plot data
    data_to_plot = {}
    
    # Add meta features first
    if "model size" in tmp:
        data_to_plot["model size"] = tmp["model size"]
        tmp.pop("model size")
    if "fps" in tmp:
        data_to_plot["fps"] = tmp["fps"]
        tmp.pop("fps")
    
    # Add individual tasks
    for k, v in tmp.items():
        if "+" not in k and "shuffle" not in k and "normal" not in k:
            data_to_plot[k] = v
    
    # Add task combinations
    for k, v in tmp.items():
        if "+" in k:
            data_to_plot[k] = v
    
    # Add shuffle/normal conditions if requested
    if args.include_conditions:
        for k, v in tmp.items():
            if "shuffle" in k or "normal" in k:
                data_to_plot[k] = v

    # Create plot
    fig_width = max(8, len(data_to_plot) * 0.5)
    plt.figure(figsize=(fig_width, 6))
    plot_relevance(data_to_plot, width=0.8)
    
    plt.ylim(0, args.ylim)
    plt.ylabel("Variance explained (R²)")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = f"rank_{args.clip_duration}_beh"
    plt.savefig(os.path.join(output_dir, f"{output_prefix}.svg"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved {output_prefix} plots to {output_dir}")

def main(args):
    if args.mode == "compute":
        compute_behaviour_task_relevance(args)
    elif args.mode == "plot":
        plot_behaviour_task_relevance(args)
    elif args.mode == "all":
        compute_behaviour_task_relevance(args)
        plot_behaviour_task_relevance(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'compute', 'plot', or 'all'")


if __name__ == "__main__":
    extra_args = [
        (("--mode",), {"type": str, "default": "plot", 
                       "choices": ["compute", "plot", "all"],
                       "help": "Mode: 'compute' to compute behaviour task relevance, 'plot' to generate plots, "
                              "'all' for compute + plots"}),
        (("--num-resample",), {"type": int, "default": 1000, 
                               "help": "Number of resamples for bootstrap confidence intervals"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots (default: script directory)"}),
        (("--ylim",), {"type": float, "default": 0.8,
                       "help": "Y-axis limit for plots"}),
        (("--use-custom-order",), {"action": "store_true",
                                   "help": "Use predefined ordering instead of sorting by values"}),
        (("--include-conditions",), {"action": "store_true",
                                     "help": "Include shuffle/normal condition data in main plot"}),
        (("--tasks",), {"type": str, "nargs": "+", "default": None,
                        "help": "List of tasks to analyze (default: predefined list)"}),
    ]
    
    args = get_args(*extra_args)
    
    # Set default tasks if not provided
    if args.tasks is None:
        args.tasks = DEFAULT_TASKS
    
    main(args)