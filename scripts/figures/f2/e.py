import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from scripts.utils import *
from scripts.utils.regions import rolls
from src.models.groups import *
from src.store import pickle_store

plt.rcParams['svg.fonttype'] = 'none'

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f2e')
os.makedirs(FIGURE_DIR, exist_ok=True)


def get_layer_region(args):
    """Compute layer-region mapping from fMRI scores"""
    FMRI_CODE = f"fmri.{args.fmri_code}.{args.inference_mode}.{args.context_duration}.{args.clip_duration}{'.hrf' if args.hrf else ''}.p"

    checked = check([FMRI_CODE])
    models = [m for m, completed in checked.items() if completed]
    models = [m for m in models if m in TEMPORAL_MODELS+['VJEPA-Temporal']]

    # print invalid models
    invalid_models = [m for m, completed in checked.items() if not completed]
    for m in invalid_models:
        print(f"{m} is not completed")

    scores = {}
    result = {}
    score_maps = {}
    for model in models:
        # FMRI
        fmri_id = f"fmri.{model}.{args.fmri_code}.{args.inference_mode}.{args.context_duration}.{args.clip_duration}{'.hrf' if args.hrf else ''}"
        fmri_scores, fmri_valid_scores, layers = score_store.load(fmri_id)

        num_layers = len(layers)

        # average over splits
        test_score = fmri_scores.mean(-1)
        valid_score = fmri_valid_scores.mean(-1)
        layer_indices = valid_score.argmax(0)
        result[model] = (layer_indices + 1) / num_layers
        whole_brain_scores = select_layer(test_score, valid_score)
        scores[model] = np.median(whole_brain_scores)
        score_maps[model] = whole_brain_scores

    # sort by score
    models = sorted(models, key=lambda x: scores[x], reverse=True)

    # print top models
    models = models[:args.topk]
    for m in models:
        print(f"{m}: {scores[m]}")

    result = {m: result[m] for m in models}
    score_maps = {m: score_maps[m] for m in models}

    return result, score_maps


def layer_consistency(layer_mapping, split=10):
    """Compute consistency of layer mapping across models"""
    layer_mapping = np.array(list(layer_mapping.values()))
    N = len(layer_mapping)
    consistency = []
    for _ in range(split):
        indices = np.random.permutation(N)
        half1 = np.mean(layer_mapping[indices[:N//2]], 0)
        half2 = np.mean(layer_mapping[indices[N//2:]], 0)
        nans = np.isnan(half1) | np.isnan(half2)
        half1, half2 = half1[~nans], half2[~nans]
        c = np.corrcoef(half1, half2)[0, 1]
        # spearman-brown correction
        c = 2 * c / (1 + c)
        consistency.append(c)
    return np.array(consistency)


def compute_layer_mapping(args):
    """Main function to compute layer mapping and generate initial plots"""
    layer_mapping, score_map = get_layer_region(args)

    # print consistency across models
    consistency = layer_consistency(layer_mapping)
    print(f"Consistency: {consistency.mean()} +/- {2*consistency.std()}")
    
    mean_layer_mapping = np.mean(list(layer_mapping.values()), 0)
    mean_score_map = np.mean(list(score_map.values()), 0)

    # plot mean layer mapping vs. score
    plt.figure(figsize=(4, 4))
    nans = np.isnan(mean_layer_mapping) | np.isnan(mean_score_map)
    plt.scatter(mean_layer_mapping[~nans], mean_score_map[~nans], s=.25, alpha=.8)
    
    from scipy.stats import pearsonr
    r, p = pearsonr(mean_layer_mapping[~nans], mean_score_map[~nans])
    plt.xlabel("Layer assignment")
    plt.ylabel("Whole-brain alignment (normalized R)")
    plt.title(f"Correlation = {r:.2f} {'' if p > 0.001 else '***'}")

    # remove spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "layer_mapping_vs_score.png"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved layer_mapping_vs_score.png to {output_dir}")

    # Store layer mapping
    cache_key = args.cache_key or "cache.tests.paper_plots.f1.layer"
    pickle_store.store(mean_layer_mapping, cache_key)
    print(f"Stored layer mapping to {cache_key}")
    
    return mean_layer_mapping


def plot_hierarchy(args):
    """Plot anatomical hierarchy vs model-mapped hierarchy"""
    cache_key = args.cache_key or "cache.tests.paper_plots.f1.layer"
    
    try:
        layer = pickle_store.load(cache_key)
    except Exception as e:
        print(f"Error loading layer data from {cache_key}: {e}")
        print("Please run with --mode compute first")
        return

    res = []
    for r in rolls.regions:
        sel = region_voxels(r)
        l = np.nanmean(layer[sel])
        res.append(l)

    plt.figure(figsize=(4.3, 4.3))
    sns.regplot(x=rolls.hier, y=res, color=HUMAN, 
                scatter_kws={"s": 17, "edgecolor": "white", "alpha": 1})
    plt.xlabel("Anatomical hierarchy")
    plt.ylabel("Model-mapped hierarchy")
    sns.despine()

    # annotate
    for i, r in enumerate(rolls.regions):
        plt.text(rolls.hier[i], res[i]+0.008, r, fontsize=8.2, ha="center", va="center")

    # plot correlation line
    from scipy.stats import pearsonr
    r, p = pearsonr(rolls.hier, res)
    print(f"Correlation: {r:.2f}, p={p:.4f}")

    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "prob_hier.svg"), bbox_inches="tight")
    plt.close()
    print(f"Saved prob_hier plots to {output_dir}")


def normalize(data):
    """Normalize data to [0, 1]"""
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def quantize(data, num_bins):
    """Quantize data into bins"""
    nans = np.isnan(data)
    bins = np.linspace(0, 1, num_bins)
    q = np.digitize(data[~nans], bins)
    q = normalize(q)
    data[~nans] = q
    return data


def plot_brain_layer(args):
    """Plot brain surface with layer mapping"""
    cache_key = args.cache_key or "cache.tests.paper_plots.f1.layer"
    
    try:
        layer = pickle_store.load(cache_key)
    except Exception as e:
        print(f"Error loading layer data from {cache_key}: {e}")
        print("Please run with --mode compute first")
        return

    color1 = args.color1 or "#186340"
    color2 = args.color2 or "#F5FFF6"

    normalized_layer = normalize(layer)
    normalized_layer = quantize(normalized_layer, args.num_bins)

    plot_dual(normalized_layer, color1, color2, with_colorbar=False)
    
    output_dir = args.output_dir or FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "layer.png"), dpi=500, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Saved layer brain plots to {output_dir}")


def plot_all(args):
    """Generate all plots"""
    plot_hierarchy(args)
    plot_brain_layer(args)


def main(args):
    if args.mode == "compute":
        compute_layer_mapping(args)
    elif args.mode == "hierarchy":
        plot_hierarchy(args)
    elif args.mode == "brain":
        plot_brain_layer(args)
    elif args.mode == "plot":
        plot_all(args)
    elif args.mode == "all":
        compute_layer_mapping(args)
        plot_all(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'compute', 'hierarchy', 'brain', 'plot', or 'all'")


if __name__ == "__main__":
    extra_args = [
        (("--mode",), {"type": str, "default": "plot", 
                       "choices": ["compute", "hierarchy", "brain", "plot", "all"],
                       "help": "Mode: 'compute' to compute layer mapping, 'hierarchy' for hierarchy plot, "
                              "'brain' for brain surface plot, 'plot' for all plots, 'all' for compute + plots"}),
        (("--topk",), {"type": int, "default": 20, 
                       "help": "Top k models to consider for layer mapping"}),
        (("--cache-key",), {"type": str, "default": None,
                           "help": "Cache key for storing/loading layer mapping (default: cache.tests.paper_plots.f1.layer)"}),
        (("--output-dir",), {"type": str, "default": None,
                            "help": "Output directory for plots (default: script directory)"}),
        (("--num-bins",), {"type": int, "default": 6,
                          "help": "Number of bins for quantizing layer data in brain plot"}),
        (("--color1",), {"type": str, "default": None,
                        "help": "Color 1 for brain plot (default: #186340)"}),
        (("--color2",), {"type": str, "default": None,
                        "help": "Color 2 for brain plot (default: #F5FFF6)"}),
    ]
    
    args = get_args(*extra_args)
    main(args)