import torch
from torch import nn
from .flags import *


class MultiTask:
    def __init__(self, dataset_names, image_steps):
        self.image_steps = image_steps
        self.dataset_names = dataset_names

    def convert_input(self, x, dataset_name):
        if dataset_name in [DATASETS.IMAGENET1K]:
            return torch.stack([x for _ in range(self.image_steps)], dim=1)
        else:
            return x

    def generate_classifiers(self, inp_dim):
        classifiers = {}
        for dataset_name in self.dataset_names:
            out_dim = OUTPUT_DIM_MAP[dataset_name]
            classifier = nn.Linear(inp_dim, out_dim)
            classifiers[dataset_name] = classifier
        classifiers = nn.ModuleDict(classifiers)
        return classifiers

    def multitask_forward(self, f, dataset_name):
        return self.classifiers[dataset_name](f)