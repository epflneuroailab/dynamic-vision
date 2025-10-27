import os
import numpy as np
import matplotlib.pyplot as plt

from scripts.utils import *
from src.analysis.regions import MAP
from src.store import pickle_store, activation_store
from src.evaluate.fmri import _normalize_assemblies
from src.evaluate.utils import time_align, time_hrf, make_stimulus_paths
from brainscore_vision.benchmark_helpers.neural_common import average_repetition

plt.rcParams['svg.fonttype'] = 'none'

MODELS = ["VJEPA-Temporal", "MotionNet", "convnext_large_imagenet_full_seed-0"]
REGIONS = ["Ventral_Stream_Visual", "Middle_Temporal_Area", "Dorsal_Stream_Visual"]
this_dir = os.path.dirname(os.path.realpath(__file__))
score_store = pickle_store.add_node("scores")
data_store = pickle_store.add_node("data")
FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f2d')
os.makedirs(FIGURE_DIR, exist_ok=True)

TARGET_DATASETS = [
    "savasegal2023-fmri-defeat",
    "savasegal2023-fmri-growth",
    "savasegal2023-fmri-iteration",
    "savasegal2023-fmri-lemonade",
]

def nanmedians(a, num):
    a_clean = np.sort(a[~np.isnan(a)])
    n = len(a_clean)
    if n == 0:
        return np.nan
    before = num // 2
    after = num - before
    return a_clean[n//2 - before:n//2 + after]

def pick_voxels(region, ceils, fmri_scores):
    nsel = MAP.get_nsel(region)

    ceiling_vals = ceils[nsel]
    fmri_vals = fmri_scores[nsel]

    # top 20% ceiling voxels
    ceil_sel = ceiling_vals > np.percentile(ceiling_vals[~np.isnan(ceiling_vals)], 80)
    median_voxel_scores = nanmedians(fmri_vals[ceil_sel], num=20)

    # get the middle 20 voxels
    voxel_ids = []
    for median_voxel_score in median_voxel_scores:
        voxel_id = np.where(fmri_scores==median_voxel_score)[0][0]
        voxel_ids.append(voxel_id)

    return voxel_ids

def select_layers(model):
    eval_id = f"fmri.{model}.12345678.block.4000.15.hrf"
    scores = score_store.load(eval_id)
    score_valid = scores[1].mean(-1)
    layers = scores[2]
    layer_ids = score_valid.argmax(0)
    return np.array(layers)[layer_ids]

def load_average_assembly(dataset, args=None):
    data_cache = data_store.add_node("cache")
    if data_cache.exists(dataset) and (args and not args.rerun_all):
        data = data_cache.load(dataset)
    else:
        assembly = data_store.load(dataset)
        assembly = average_repetition(assembly)
        data_cache.store(assembly, dataset)
        data = assembly
    return data

def make_predictions(model, voxel_id, layer, target_datasets, time_limit):
    train_activations = []
    train_assemblies = []
    test_activations = []
    test_assemblies = []
    for dataset in target_datasets:
        activation_id = f"{model}.{dataset}.block.4000"
        try:
            activation = activation_store.load(activation_id)
        except ValueError:
            print(f"Please run activation extraction for {activation_id} first.")
            exit(1)
        activation = activation.isel(neuroid=activation.layer==layer)
        assembly = load_average_assembly(dataset)
        assembly = make_stimulus_paths({dataset: assembly})[dataset]

        # select voxel
        assembly = assembly.isel(neuroid=[voxel_id])
        
        # normalize assembly
        assembly = _normalize_assemblies({dataset: assembly})[dataset]

        # time align
        activation, assembly = time_align(activation, assembly)
        if dataset not in ["mcmahon2023-fmri", "lahner2024-fmri"]:
            activation = time_hrf(activation, tr=1) 

        # split train and test
        if dataset in target_datasets:
            time_end = activation["time_bin_end"].values[-1]
            train_activations.append(activation.isel(time_bin=activation.time_bin_end <= (time_end-time_limit)))
            test_activations.append(activation.isel(time_bin=activation.time_bin_end > (time_end-time_limit)))
            train_assemblies.append(assembly.isel(time_bin=assembly.time_bin_end <= (time_end-time_limit)))
            test_assemblies.append(assembly.isel(time_bin=assembly.time_bin_end > (time_end-time_limit)))
        else:
            train_activations.append(activation)
            train_assemblies.append(assembly)

    def _take_values(asm):
        arr = asm.transpose('time_bin', ..., 'neuroid').values
        num_neurons = arr.shape[-1]
        return arr.reshape(-1, num_neurons)

    train_x = []
    test_x = []
    train_y = []
    test_y = []
    for activation, assembly in zip(train_activations, train_assemblies):
        train_x.append(_take_values(activation))
        train_y.append(_take_values(assembly))

    for activation, assembly in zip(test_activations, test_assemblies):
        test_x.append(_take_values(activation))
        test_y.append(_take_values(assembly))

    train_x = np.concatenate(train_x, 0)
    train_y = np.concatenate(train_y, 0)

    # set ridgecv
    from sklearn.linear_model import RidgeCV
    model_ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
    model_ridge.fit(train_x, train_y)

    ret = []
    for x, y, dataset, assembly in zip(test_x, test_y, target_datasets, test_assemblies):
        prediction = model_ridge.predict(x)
        corr_score = np.corrcoef(y.flatten(), prediction.flatten())[0, 1]
        print(f"Correlation score: {corr_score:.4f}; Layer: {layer}, Voxel ID: {voxel_id}")
        ret.append(
            {
                "voxel_id": voxel_id,
                "dataset": dataset,
                "stimuli": assembly.stimulus_path.values,
                "time_start": assembly.time_bin_start.values[0],
                "time_end": assembly.time_bin_end.values[-1],
                "BOLD": y.flatten(), 
                "prediction": prediction.flatten(),
                "correlation": corr_score,
            }
        )

    return ret


def run_predictions(args):
    """Generate predictions and store results"""
    time_limit = 240 * 1000
    predictions = {}

    ceiling = get_ceiling(args)
    fmri, models = collect(args, types=['fmri'], exclude_pixels=True)
    
    models_to_run = MODELS if args.all_models else [MODELS[0]]
    
    for model in models_to_run:
        fmri_scores = fmri[model].mean(0)
        fmri_scores = select_and_ceil(fmri_scores, ceiling)
        
        selected_layers = select_layers(model)

        rets = {}
        for region in REGIONS:
            region_name = region
            if region in MAP.regions: 
                region = MAP.regions[region]
            voxel_ids = pick_voxels(region, ceiling, fmri_scores)
            for voxel_id in voxel_ids:
                layer = selected_layers[voxel_id]
                ret = make_predictions(model, voxel_id, layer, TARGET_DATASETS, time_limit)
                rets.setdefault(region_name, []).append(ret)

        predictions[model] = rets

    output_file = args.output_file or "cache.tests.paper_plots.f1.predictions_long_video"
    pickle_store.store(predictions, output_file)
    print(f"Predictions saved to: {output_file}")


def plot_predictions(args):
    """Load predictions and generate plots"""
    input_file = args.input_file or "cache.tests.paper_plots.f1.predictions_long_video"
    
    # Load predictions
    predictions = pickle_store.load(input_file)
    
    color_true = "#27A169"
    color_pred = "#1D65C3"
    interval = 10

    models_to_plot = MODELS if args.all_models else [MODELS[0]]
    
    # Plot individual model predictions
    for model in models_to_plot:
        if model not in predictions:
            print(f"Model {model} not found in predictions")
            continue
            
        fig, axs = plt.subplots(figsize=(8, 1.5*3), nrows=len(REGIONS), ncols=4)
        prediction = predictions[model]
        
        for r, region in enumerate(REGIONS):
            region_data = prediction.get(region, [])
            # Just plot the first voxel in each region for example
            voxel_data = region_data[0]

            # Plot predictions on all datasets
            for i, pred_data in enumerate(voxel_data):
                    
                y = pred_data['BOLD']
                pred = pred_data['prediction']

                ax = axs[r][i]
                xs = np.arange(len(y))
                ax.plot(xs, y, color=color_true, linewidth=1.2, label='True' if r==0 and i==0 else '')
                ax.plot(xs, pred, color=color_pred, linewidth=1.2, label='Predicted' if r==0 and i==0 else '')

                # Styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if i != 0:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([])
                else:
                    ax.spines['left'].set_position(('outward', interval))
                    ax.set_ylabel("z-score")

                ax.set_ylim(-5, 5)
                ax.set_xlim(0, len(y))
                ax.set_xticks([0, len(y)])

                # Add correlation
                corr = np.corrcoef(y, pred)[0, 1]
                ax.text(0.4, 0.05, f"R={corr:.2f}", transform=ax.transAxes, fontsize=8, ha='right')

        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        
        output_prefix = args.output_prefix or model
        plt.savefig(f"{FIGURE_DIR}/{output_prefix}_predictions.svg", bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved {output_prefix}_predictions.svg")

    # Generate correlation scatter plot
    if args.plot_correlation and len(models_to_plot) >= 2:
        ys = []
        preds = {}
        
        for m, model in enumerate(models_to_plot[:2]):
            if model not in predictions:
                continue
            preds[model] = []
            prediction = predictions[model]
            
            for r, region in enumerate(REGIONS):
                region_data = prediction.get(region, [])
                for pred_data in region_data:
                    if isinstance(pred_data, list) and len(pred_data) > 0:
                        y = pred_data[0]['BOLD']
                        pred = pred_data[0]['prediction']
                        if m == 0:
                            ys.append(y)
                        preds[model].append(pred)

        if ys:
            y = np.concatenate(ys)
            for model in preds:
                preds[model] = np.concatenate(preds[model])

            plt.figure(figsize=(4, 4))
            for model in models_to_plot[:2]:
                if model in preds:
                    corr = np.corrcoef(y, preds[model])[0, 1]
                    print(f"{model}: R={corr:.4f}")
                    plt.scatter(y, preds[model], label=model, s=10, alpha=0.25, edgecolor='none')

            plt.xlim(-3.5, 3.5)
            plt.ylim(-3.5, 3.5)
            plt.legend()
            plt.xlabel("True")
            plt.ylabel("Predicted")
            
            output_prefix = args.output_prefix or "correlation"
            plt.savefig(f"{FIGURE_DIR}/{output_prefix}_correlation.svg", bbox_inches='tight', transparent=True)
            plt.close()
            print(f"Saved {output_prefix}_correlation.svg")


def main(args):
    if args.mode == "predict":
        run_predictions(args)
    elif args.mode == "plot":
        plot_predictions(args)
    elif args.mode == "both":
        run_predictions(args)
        plot_predictions(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'predict', 'plot', or 'both'")


if __name__ == "__main__":
    extra_args = [
        ("--mode", {"type": str, "default": "plot", "choices": ["predict", "plot", "both"],
                    "help": "Mode: 'predict' to generate predictions, 'plot' to generate plots, 'both' for both"}),
        ("--all-models", {"action": "store_true", "help": "Run/plot all models instead of just the first one"}),
        ("--input-file", {"type": str, "default": None, 
                          "help": "Input file for plotting (default: cache.tests.paper_plots.f1.predictions_long_video)"}),
        ("--output-file", {"type": str, "default": None,
                           "help": "Output file for predictions (default: cache.tests.paper_plots.f1.predictions_long_video)"}),
        ("--output-prefix", {"type": str, "default": None,
                            "help": "Prefix for output plot files (default: model name or 'correlation')"}),
        ("--plot-correlation", {"action": "store_true", "help": "Generate correlation scatter plot"}),
        ("--rerun-all", {"action": "store_true", "help": "Rerun all computations without using cache"}),
    ]
    
    args = get_args(*extra_args)
    main(args)