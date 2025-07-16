import argparse
import os
from src.models.groups import *
from src.store import pickle_store

score_store = pickle_store.add_node("scores")


def check(codes, models=ALL_MODELS):
    ret = {}
    filenames = os.listdir(score_store.root)
    for model in models:
        if model in AVOID: continue
        all_completed = True
        for code in codes:
            tmp = code.split('.')
            type = tmp[0]
            indices = tmp[1]
            others = tmp[2:]
            constraint = "." if not others else "."+".".join(others)

            if type == 'fmri':
                eval_id_start = f"{type}.{model}.{indices}{constraint}"
                completed = False
                for filename in filenames:
                    if filename.startswith(eval_id_start):
                        completed = True
                        break

            else:
                completed = True
                ids = indices if "/" not in indices else indices.split("/")
                for id in ids:
                    completed_ = False
                    eval_id_start = f"{type}.{model}.{id}{constraint}"
                    for filename in filenames:
                        if filename.startswith(eval_id_start):
                            completed_ = True
                            break
                    completed = completed and completed_

            all_completed = all_completed and completed
        ret[model] = all_completed

    return ret