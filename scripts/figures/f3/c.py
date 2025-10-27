import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.utils import *
from src.store import pickle_store

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f3c')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Default tasks to analyze
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

# Subtraction patterns to analyze
DEFAULT_SUBTRACTION = [
    ['imagenet2012'],
    ['imagenet2012', 'afd2022'],
    ['imagenet2012', 'hdm05'],
]


def compute_task_relevance(args):
    """Compute task relevance statistics using regression and permutation analysis"""
    print("Computing task relevance statistics...")
    
    ceiling = get_ceiling(args)
    data, task, meta, models = collect(args, types=[args.type, 'task', 'meta'], exclude_pixels=True)
    
    # Prepare data scores
    if args.type == 'fmri':
        data_scores = np.array([data[m].mean(0) for m in models])
        data_scores = select_and_ceil(data_scores, ceiling, "Whole_Brain").reshape(-1, 1)
    elif args.type == 'elec':
        data_scores = np.array([data[m].mean(0) for m in models])
    else:
        raise ValueError(f"Unsupported type: {args.type}")
    
    # Prepare task scores and meta values
    task_scores = np.array([[task[t][m].mean(0) for m in models] for t in args.tasks]).T
    meta_values = np.array([list(meta[m].values()) for m in models])

    results = {}

    def _get_stat_r2(name, input_scores):
        """Compute R² with confidence intervals using resampling"""
        _, r2 = resample_analyze(input_scores, data_scores, n_resamples=args.num_resample)
        mean, ci_l, ci_u = r2
        print(f"{name}: {mean.item():.4f} ({ci_l.item():.4f}, {ci_u.item():.4f})")
        return r2

    def _get_stat_r2_remained(names, input_scores, sub_scores):
        """Compute R² with subtraction using permutation testing"""
        r2s = permutation_analyze(input_scores, data_scores, *sub_scores, 
                                 n_permutations=args.num_permutations)
        for r2, name in zip(r2s, names):
            mean, pval = r2
            sig = '*' if pval < 0.05 else 'n.s.'
            print(f"{name}: {mean.item():.4f} {sig}")
        return r2s

    # Meta features (fps, model size)
    meta_names = ['fps', 'model size']
    for i, meta_name in enumerate(meta_names):
        results[meta_name] = _get_stat_r2(meta_name, meta_values[:, i].reshape(-1, 1))
    
    # Individual tasks
    print("\nIndividual tasks:")
    for i, task_name in enumerate(args.tasks):
        results[task_name] = _get_stat_r2(task_name, task_scores[:, i].reshape(-1, 1))

    # Task combinations
    print("\nTask combinations:")
    task_combinations = [
        ('imagenet2012 + afd2022', ['imagenet2012', 'afd2022']),
        ('imagenet2012 + hdm05', ['imagenet2012', 'hdm05']),
        ('imagenet2012 + ding2012', ['imagenet2012', 'ding2012']),
    ]
    
    for combo_name, combo_tasks in task_combinations:
        if all(t in args.tasks for t in combo_tasks):
            task_ids = [args.tasks.index(t) for t in combo_tasks]
            results[combo_name] = _get_stat_r2(combo_name, task_scores[:, task_ids])

    # Task subtraction analysis
    print("\nTask subtraction analysis:")
    for i, task_name in enumerate(args.tasks):
        input_scores = task_scores[:, [i]]
        sub_scores = []
        names = [task_name]
        
        for to_subtract in args.subtraction:
            if all(t in args.tasks for t in to_subtract):
                to_subtract_ids = [args.tasks.index(t) for t in to_subtract]
                sub_score = task_scores[:, to_subtract_ids]
                sub_scores.append(sub_score)
                names.append(f"{task_name} - {'+'.join(to_subtract)}")
        
        if sub_scores:  # Only compute if there are subtractions to perform
            results[f"sub:{task_name}"] = _get_stat_r2_remained(names, input_scores, sub_scores)

    # All tasks subtraction
    print("\nAll tasks subtraction:")
    input_scores = task_scores
    sub_scores = []
    names = ["all"]
    
    for to_subtract in args.subtraction:
        if all(t in args.tasks for t in to_subtract):
            to_subtract_ids = [args.tasks.index(t) for t in to_subtract]
            sub_score = task_scores[:, to_subtract_ids]
            sub_scores.append(sub_score)
            names.append(f"all - {'+'.join(to_subtract)}")

    if sub_scores:
        results["sub:all"] = _get_stat_r2_remained(names, input_scores, sub_scores)

    # Store results
    cache_key = f"cache.tests.paper_plots.f2.{args.clip_duration}"
    pickle_store.store(results, cache_key)
    print(f"\nStored results to {cache_key}")
    
    return results


def plot_relevance(d, **kwargs):
    """Plot horizontal bar chart with error bars"""
    names = list(d.keys())
    means = [v[0] for v in d.values()]
    cis = [(v[1], v[2]) for v in d.values()]
    
    # Horizontal bar plot
    plt.barh(names, means, **kwargs)

    # Add error bars
    for i, (ci_low, ci_high) in enumerate(cis):
        ci_low = max(ci_low, 0)
        plt.plot([ci_low, ci_high], [i, i], color="black")

    # Print statistics
    for name, mean in zip(names, means):
        print(f"{name}: {mean:.3f} ({cis[names.index(name)][0]:.3f}, {cis[names.index(name)][1]:.3f})")

    plt.xlim(0)
    sns.despine()


def plot_task_relevance(args):
    """Generate task relevance ranking plot"""
    print("Generating task relevance plot...")
    
    # Load cached data
    cache_key = f"cache.tests.paper_plots.f2.{args.clip_duration}"
    try:
        data = pickle_store.load(cache_key)
    except Exception as e:
        print(f"Error loading cached data: {e}")
        print("Please run with --mode compute first")
        return

    # Sort and organize data for plotting
    tmp = sorted({k: v for k, v in data.items() if "sub" not in k}.items(), 
                 key=lambda item: item[1][0].item(), reverse=False)
    tmp = {k: [a.item() for a in v] for k, v in tmp}
    
    # Organize plot data
    data_to_plot = {
        "model size": tmp["model size"],
        "fps": tmp["fps"],
    }
    tmp.pop("model size")
    tmp.pop("fps")
    
    # Add individual tasks
    data_to_plot = {
        **data_to_plot,
        **{k: v for k, v in tmp.items() if "+" not in k},
    }
    
    # Add task combinations at the end
    if "imagenet2012 + ding2012" in tmp:
        data_to_plot["imagenet2012 + ding2012"] = tmp["imagenet2012 + ding2012"]
    if "imagenet2012 + hdm05" in tmp:
        data_to_plot["imagenet2012 + hdm05"] = tmp["imagenet2012 + hdm05"]
    if "imagenet2012 + afd2022" in tmp:
        data_to_plot["imagenet2012 + afd2022"] = tmp["imagenet2012 + afd2022"]

    # Create plot
    plt.figure(figsize=(3, 7.5))
    plot_relevance(data_to_plot, height=0.65)

    # Invert y-axis to show highest at top
    plt.gca().invert_yaxis()
    
    plt.xlabel("Variance explained (R²)")
    plt.ylabel("")

    # Save plot
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    output_prefix = f"rank_{args.clip_duration}"
    plt.savefig(os.path.join(output_dir, f"{output_prefix}.svg"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved {output_prefix} plots to {output_dir}")

def main(args):
    if args.mode == "compute":
        compute_task_relevance(args)
    elif args.mode == "plot":
        plot_task_relevance(args)
    elif args.mode == "all":
        compute_task_relevance(args)
        plot_task_relevance(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'compute', 'plot', or 'all'")


if __name__ == "__main__":
    extra_args = [
        (("--mode",), {"type": str, "default": "plot", 
                       "choices": ["compute", "plot", "all"],
                       "help": "Mode: 'compute' to compute task relevance, 'plot' to generate plots, "
                              "'all' for compute + plots"}),
        (("--num-resample",), {"type": int, "default": 1000, 
                               "help": "Number of resamples for bootstrap confidence intervals"}),
        (("--num-permutations",), {"type": int, "default": 1000, 
                                   "help": "Number of permutations for significance testing"}),
        (("--type",), {"type": str, "default": "fmri", 
                       "choices": ["fmri", "elec"],
                       "help": "Type of neural data to analyze"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots (default: script directory)"}),
        (("--tasks",), {"type": str, "nargs": "+", "default": None,
                        "help": "List of tasks to analyze (default: predefined list)"}),
    ]
    
    args = get_args(*extra_args)
    
    # Set default tasks if not provided
    if args.tasks is None:
        args.tasks = DEFAULT_TASKS
    
    # Set default subtraction patterns
    args.subtraction = DEFAULT_SUBTRACTION
    
    main(args)