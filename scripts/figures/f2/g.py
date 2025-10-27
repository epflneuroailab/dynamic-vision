import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from scripts.utils import *
from src.models.groups import *
from src.store import pickle_store
from src.data import data_store, BEHAVIOURS, STATIC
from src.evaluate.behaviour.i1i2 import metrics

plt.rcParams['svg.fonttype'] = 'none'

TASK_OF_INTEREST = ['imagenet2012', 'afd2022']
BENCHMARK_OF_INTEREST = [
    "han2024-RGB",
    "han2024-RGB-S",
    "han2024-J-6P",
    "han2024-J-6P-S"
]
BENCHMARK_OF_INTEREST_NAMES = [
    "Normal",
    "Shuffled",
    "Normal",
    "Shuffled"
]
PLOT_TOP_K = 10
SCALE = 0.7

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f2g')
os.makedirs(FIGURE_DIR, exist_ok=True)


def compute_error_stats(args):
    """Compute error pattern statistics from behaviour and task benchmarks"""
    print("Computing error pattern statistics...")
    
    beh, task, models = collect(args, types=['beh', 'task'], 
                                models=[m for m in ALL_MODELS if m not in CONTROLS])
    beh_benchmarks = list(beh.keys())
    task_benchmarks = list(task.keys())

    # Compute human ceiling
    beh_ceiling = {}
    for dataset in beh_benchmarks:
        data = data_store.load(dataset)
        o2 = metrics["O2"]
        ceiling = o2.ceiling(data)
        beh_ceiling[dataset] = ceiling
        print(f"{dataset} ceiling: {ceiling}")

    # Compute task scores
    task_scores = {} 
    for benchmark, data in task.items():
        task_score = np.array([np.mean(data[m]) for m in models])
        task_scores[benchmark] = task_score

    # Compute behaviour scores
    beh_scores = {}
    for benchmark, data in beh.items():
        beh_scores[benchmark] = []
        for m in models:
            beh_score, human_acc, model_acc = data[m]
            beh_score = np.nanmean(beh_score)
            model_acc_mean = np.nanmean(model_acc)
            human_acc_mean = np.nanmean(human_acc)
            beh_scores[benchmark].append((beh_score, model_acc_mean, human_acc_mean))
        beh_scores[benchmark] = np.array(beh_scores[benchmark])  # (n_models, 3)

    # Discard nan models
    nans = np.array([np.isnan(beh_scores[b]).any(axis=1) for b in beh_benchmarks]).any(axis=0)
    models_clean = [m for i, m in enumerate(models) if not nans[i]]
    
    for b in beh_benchmarks:
        beh_scores[b] = beh_scores[b][~nans]
    for t in task_benchmarks:
        task_scores[t] = task_scores[t][~nans]

    print()
    for i, m in enumerate(models):
        if nans[i]:
            print(f"Discarded due to nan: {m}")

    # Collect error pattern stats
    error_stats = [task_scores, {b: beh_scores[b][:, 0] for b in beh_benchmarks}, beh_ceiling]

    # Collect acc pattern stats
    model_acc = {b: beh_scores[b][:, 1] for b in beh_benchmarks}
    human_acc = {b: beh_scores[b][:, 2] for b in beh_benchmarks}
    acc_stats = [model_acc, human_acc]

    # Store results
    pickle_store.store((error_stats, models_clean), "cache.tests.paper_plots.f5.err_stats")
    pickle_store.store((acc_stats, models_clean), "cache.tests.paper_plots.f5.acc_stats")
    
    print(f"Stored error stats for {len(models_clean)} models")
    
    return error_stats, acc_stats, models_clean


def plot_error_patterns(args):
    """Plot error pattern alignment across models and benchmarks"""
    print("Generating error pattern plots...")
    
    # Load cached data
    try:
        (task_scores, beh_scores, beh_ceiling), models = pickle_store.load("cache.tests.paper_plots.f5.err_stats")
        (model_acc, human_acc), models = pickle_store.load("cache.tests.paper_plots.f5.acc_stats")
    except Exception as e:
        print(f"Error loading cached data: {e}")
        print("Please run with --mode compute first")
        return

    # Filter task scores
    task_scores = {k: v for k, v in task_scores.items() if k in TASK_OF_INTEREST}
    
    model_err = beh_scores
    benchmarks = list(model_acc.keys())
    B = len(BENCHMARK_OF_INTEREST)
    M = len(model_acc[list(model_acc.keys())[0]])
    tasks = list(task_scores.keys())
    T = len(tasks)
    
    aver_acc_scores = np.mean([model_acc[b] for b in BENCHMARK_OF_INTEREST], axis=0)
    aver_beh_scores = np.mean([beh_scores[b] for b in beh_scores], axis=0)
    best_model_idx = np.argmax(aver_acc_scores)
    best_model = models[best_model_idx]
    model_colors = [colors.get_color_by_model(m) for m in models]
    
    print(f"Best model: {best_model}")

    # Plot all models error patterns
    fig, ax = plt.subplots(figsize=(6.2*SCALE, 4.6*SCALE))
    all_models_err_vals = np.array([[model_err[b][i] for b in BENCHMARK_OF_INTEREST] 
                                    for i in range(M)]).T  # [B, M]
    xs_wiggle = np.random.normal(0, 0.05, all_models_err_vals.shape)
    
    # Normalize to [0, 1]
    ceiled_all_models_err_vals = all_models_err_vals / np.max(all_models_err_vals, axis=1, keepdims=True)
    aver_models_err_vals = np.mean(ceiled_all_models_err_vals, axis=0)
    top_models = np.argsort(aver_models_err_vals)[::-1]
    
    print("\nModel error pattern alignment:")
    for i, m in enumerate(top_models[:PLOT_TOP_K]):
        x = all_models_err_vals[:, m]
        y = np.arange(B) + xs_wiggle[:, m]
        sns.lineplot(y=y, x=x, c=colors.get_color_by_model(models[m]), 
                    ax=ax, alpha=1, linewidth=.8, orient='y', zorder=-100)

    for b in range(B):
        plt.scatter(y=b+xs_wiggle[b], x=all_models_err_vals[b], c=model_colors)
        human_ceil = beh_ceiling[BENCHMARK_OF_INTEREST[b]].values
        human_mean = np.mean(human_ceil)
        human_std = np.std(human_ceil) * 1.96 / np.sqrt(len(human_ceil))  # 95% CI
        plt.fill_betweenx([b-0.5, b+0.5], human_mean-human_std, human_mean+human_std, 
                         color="gray", alpha=0.5, edgecolor=None, zorder=-200)

    # Print model ordering for each benchmark
    for b in range(B):
        model_ordering = np.argsort(all_models_err_vals[b])[::-1]
        print(f"\n{BENCHMARK_OF_INTEREST[b]} ordering:")
        print(f"  {', '.join([models[m] for m in model_ordering[:5]])}...")

    sns.despine()
    plt.xlabel("Human error pattern alignment")
    plt.yticks(np.arange(B), BENCHMARK_OF_INTEREST_NAMES)
    plt.title(args.plot_title or "Biological motion")
    ax.invert_yaxis()
    plt.ylim(B-0.5, -0.5)
    plt.xlim(0)
    
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "all_models_err.svg"), bbox_inches='tight')
    plt.close()
    
    print(f"Saved all_models_err plots to {output_dir}")


def plot_comparison(args):
    """Plot comparison of models across different metrics"""
    print("Generating comparison plots...")
    
    try:
        (task_scores, beh_scores, beh_ceiling), models = pickle_store.load("cache.tests.paper_plots.f5.err_stats")
        (model_acc, human_acc), models = pickle_store.load("cache.tests.paper_plots.f5.acc_stats")
    except Exception as e:
        print(f"Error loading cached data: {e}")
        print("Please run with --mode compute first")
        return
    
    # Filter task scores
    task_scores = {k: v for k, v in task_scores.items() if k in TASK_OF_INTEREST}
    
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot task performance vs behaviour alignment
    fig, ax = plt.subplots(figsize=(6, 5))
    
    aver_task_scores = np.mean([task_scores[t] for t in task_scores], axis=0)
    aver_beh_scores = np.mean([beh_scores[b] for b in BENCHMARK_OF_INTEREST], axis=0)
    model_colors = [colors.get_color_by_model(m) for m in models]
    
    plt.scatter(aver_task_scores, aver_beh_scores, c=model_colors, s=50, alpha=0.7)
    
    # Add correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(aver_task_scores, aver_beh_scores)
    plt.title(f"Task performance vs behaviour alignment (r={r:.2f}, p={p:.3f})")
    plt.xlabel("Average task performance")
    plt.ylabel("Average behaviour alignment")
    sns.despine()
    
    plt.savefig(os.path.join(output_dir, "task_vs_behaviour.svg"), bbox_inches='tight')
    plt.close()
    
    print(f"Saved task_vs_behaviour plots to {output_dir}")


def main(args):
    if args.mode == "compute":
        compute_error_stats(args)
    elif args.mode == "plot":
        plot_error_patterns(args)
        if args.include_comparison:
            plot_comparison(args)
    elif args.mode == "all":
        compute_error_stats(args)
        plot_error_patterns(args)
        if args.include_comparison:
            plot_comparison(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'compute', 'plot', or 'all'")


if __name__ == "__main__":
    extra_args = [
        (("--mode",), {"type": str, "default": "plot", 
                       "choices": ["compute", "plot", "all"],
                       "help": "Mode: 'compute' to compute error stats, 'plot' to generate plots, "
                              "'all' for compute + plots"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots (default: script directory)"}),
        (("--plot-title",), {"type": str, "default": None,
                            "help": "Title for the error pattern plot (default: 'Biological motion')"}),
        (("--include-comparison",), {"action": "store_true",
                                     "help": "Include comparison plots of task vs behaviour"}),
        (("--plot-top-k",), {"type": int, "default": PLOT_TOP_K,
                            "help": "Number of top models to highlight in plots"}),
    ]
    
    args = get_args(*extra_args)
    
    # Update global PLOT_TOP_K if specified
    if args.plot_top_k != PLOT_TOP_K:
        globals()['PLOT_TOP_K'] = args.plot_top_k
    
    main(args)