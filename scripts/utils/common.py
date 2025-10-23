import argparse
import os
import numpy as np
from src.store import pickle_store
from src.check import check
from src.analysis.permutation import permutation_analyze
from src.analysis.regression import resample_analyze
from src.analysis.regions import get_region_voxels, abbreviation_map, cortical_divisions
from src.analysis.fdr import false_discovery_control
from src.analysis.search_light import generate_power_adj
from src.ceiling import compute_joint_ceiling_by_codes
from src.data import TASKS, ELECTRODES, BEHAVIOURS
from src.models.groups import *

import matplotlib.pyplot as plt; plt.rcParams['svg.fonttype'] = 'none'


score_store = pickle_store.add_node("scores")
control_store = pickle_store.add_node("control")

def pval_to_stars(pval):
    if pval > 0.05:
        return "n.s."
    elif pval > 0.01:
        return "*"
    elif pval > 0.001:
        return "**"
    else:
        return "***"

def get_ceiling(args):
    ceiling = compute_joint_ceiling_by_codes(args.fmri_code, threshold=args.ceiling_threshold)
    return ceiling

def region_voxels(region):
    if isinstance(region, (list, tuple)):
        region = [abbreviation_map.get(r, r) for r in region]
    else:
        region = abbreviation_map.get(region, region)
    return get_region_voxels(region)

def select_and_ceil(fmri_scores, ceiling, region=None):
    if isinstance(region, (list, tuple)):
        region = [abbreviation_map.get(r, r) for r in region]
    elif region is not None:
        region = abbreviation_map.get(region, region)
    else:
        region = None

    fmri_scores[..., np.isnan(ceiling)] = np.nan
    c = ceiling.copy()
    if region == "Whole_Brain":
        return np.nanmedian(fmri_scores, axis=-1) / np.nanmedian(c, axis=-1)
    elif region is None:
        return fmri_scores / c
    else:
        voxels = get_region_voxels(region)
        fmri_scores = fmri_scores[..., voxels]
        c = c[..., voxels]
        return np.nanmedian(fmri_scores, axis=-1) / np.nanmedian(c, axis=-1)

def select_layer(scores, valid_scores, top_layers=None, min_top_layers=2):
    if top_layers is not None:
        num_layers = scores.shape[0]
        target_num_layers = int(num_layers * top_layers)
        target_num_layers = max(target_num_layers, min_top_layers)
        scores = scores[-target_num_layers:]
        valid_scores = valid_scores[-target_num_layers:]

    layer_indices = valid_scores.argmax(0)
    if len(scores.shape) == 1:
        return scores[layer_indices]

    return scores[layer_indices, range(scores.shape[1])]

def aggregate_output_dim(data_name, data, mode="mean"):
    if data_name == "selfmotion":
        # we don't want pitch and yaw
        data = data[..., [0,1,2,3,6]]

    if len(data.shape) == 2:
        if mode == "mean":
            return data.mean(-1)
        elif mode == "median":
            return np.median(data, -1)
    return data

def collect(args, types=['fmri', 'task', 'elec', 'beh', 'meta'], models=ALL_MODELS, exclude_pixels=True, silent=False):
    CODES = {
        'fmri': f"fmri.{args.fmri_code}.{args.inference_mode}.{args.context_duration}.{args.clip_duration}{'.hrf' if args.hrf else ''}.p",
        'task': f"task.{'/'.join(args.task_code)}",
        'elec': f"elec.{args.elec_code}.{args.inference_mode}",
        'beh': f"behaviour.{args.beh_code}.{args.inference_mode}.{args.context_duration}",
    }

    codes = [CODES[t] for t in types if t != 'meta']
    checked = check(codes, models=models)
    models = [m for m, completed in checked.items() if completed]
    if exclude_pixels:
        models = [m for m in models if 'pixels' != m]

    # print invalid models
    invalid_models = [m for m, completed in checked.items() if not completed]
    if not silent:
        for m in invalid_models: print(f"{m} is not completed")

    ret = {}
    if 'fmri' in types:
        fmri = {}
        for model in models:
            # FMRI
            fmri_id = f"fmri.{model}.{args.fmri_code}.{args.inference_mode}.{args.context_duration}.{args.clip_duration}{'.hrf' if args.hrf else ''}"
            fmri_scores, fmri_valid_scores, layers = score_store.load(fmri_id)

            # average over splits
            test_scores = []
            for split in range(fmri_scores.shape[-1]):
                test_score = fmri_scores[..., split]
                valid_score = fmri_valid_scores[..., split]
                test_scores.append(select_layer(test_score, valid_score))
            test_scores = np.stack(test_scores, 0)
            fmri[model] = test_scores
        ret['fmri'] = fmri

    if 'task' in types:
        tasks = {}
        for model in models:
            # TASK
            for task_code in args.task_code:
                task = list(TASKS.keys())[int(task_code)]
                task_id = f"task.{model}.{task_code}.{args.inference_mode}.{args.context_duration}"
                task_scores, task_valid_scores, layers = score_store.load(task_id)
                test_scores = []
                for split in range(task_scores.shape[-1]):
                    test_score = task_scores[..., split]
                    valid_score = task_valid_scores[..., split]
                    test_score, valid_score = aggregate_output_dim(task, test_score), aggregate_output_dim(task, valid_score)
                    test_scores.append(select_layer(test_score, valid_score, args.top_layers, args.min_top_layers))
                test_score = np.stack(test_scores, 0)
                tasks.setdefault(task, {}).setdefault(model, test_score)
        ret['task'] = tasks

    if 'elec' in types:
        # ELEC
        elec = {}
        for model in models:
            for elec_code in args.elec_code:
                elec_name = ELECTRODES[int(elec_code)]
                elec_id = f"elec.{model}.{elec_code}.{args.inference_mode}"
                elec_scores, elec_valid_scores, layers = score_store.load(elec_id)
                test_scores = []
                for split in range(elec_scores.shape[-1]):
                    test_score = elec_scores[..., split]
                    valid_score = elec_valid_scores[..., split]
                    test_score, valid_score = aggregate_output_dim(elec_name, test_score, "median"), aggregate_output_dim(elec_name, valid_score, "median")
                    test_scores.append(select_layer(test_score, valid_score))
                test_score = np.stack(test_scores, 0)
                elec.setdefault(elec_name, {}).setdefault(model, test_score)
        ret['elec'] = elec

    if 'beh' in types:
        # BEHAVIOUR
        beh = {}
        for model in models:
            for beh_code in args.beh_code:
                beh_name = list(BEHAVIOURS.keys())[int(beh_code)]
                beh_id = f"behaviour.{model}.{beh_code}.{args.inference_mode}.{args.context_duration}"
                (beh_scores, human_acc, model_accs), layers = score_store.load(beh_id)
                model_acc_scores = np.mean(model_accs, 1)
                layer = model_acc_scores.argmax()  # here we select the layer with the highest testing accuracy
                model_accs = model_accs[layer]
                test_scores = []
                for split in range(beh_scores.shape[-1]):
                    score = beh_scores[..., split]
                    test_scores.append(score[layer])
                test_score = np.stack(test_scores, 0)
                beh.setdefault(beh_name, {}).setdefault(model, (test_score, human_acc, model_accs))
        ret['beh'] = beh

    if 'meta' in types:
        meta = {}
        for model in models:
            meta_id = f"{model}"
            meta[model] = control_store.load(meta_id)
        ret['meta'] = meta

    ret = [ret[t] for t in types]

    return *ret, models
                        
def get_args(*extra_args):
    parser = argparse.ArgumentParser(description="Score a model on a task")
    parser.add_argument("--task-code", type=str, default="0/1/2/3/4/5/6/7/8/9/10", help="Tasks to perform correlation analysis")
    parser.add_argument("--fmri-code", type=str, default="12345678", help="FMRI code to perform correlation analysis")
    parser.add_argument("--elec-code", type=str, default="0123456", help="electrode codes to perform correlation analysis")
    parser.add_argument("--beh-code", type=str, default="345678", help="bebaviour codes to perform correlation analysis")
    parser.add_argument("--context-duration", type=float, default=4000, help="Duration of the context window")
    parser.add_argument("--inference-mode", type=str, default="block", help="Inference mode")
    parser.add_argument("--no-hrf", action="store_true", help="Use HRF")
    parser.add_argument("--top-layers", type=float, default=0.2, help="Top layers (percentage) to consider")
    parser.add_argument("--min-top-layers", type=int, default=2, help="Minimum number of top layers to consider")
    parser.add_argument("--num-perm", type=int, default=1, help="Number of iterations for permutation test")
    parser.add_argument("--clip-duration", type=int, default=15, help="Duration of the clip (sec) for evaluation stage")
    parser.add_argument("--ceiling-threshold", type=float, default=0.4, help="Threshold for ceiling computation")
    for args, kwargs in extra_args:
        if not isinstance(args, (tuple, list)):
            args = (args, )
        parser.add_argument(*args, **kwargs)
    args = parser.parse_args()
    args.hrf = not args.no_hrf

    if '/' in args.task_code:
        args.task_code = args.task_code.split('/')
    if '/' in args.elec_code:
        args.elec_code = args.elec_code.split('/')
    if '/' in args.fmri_code:
        args.fmri_code = args.fmri_code.split('/')

    return args
