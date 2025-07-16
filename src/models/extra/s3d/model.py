import torch
import os
import numpy as np
from torchvision import transforms
from s3dg_howto100m import S3D

from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file

import torch
import torch.nn as nn
from torchvision.models.video.s3d import S3D as torch_S3D

from ..utils import MultiTask


class S3D(torch_S3D, MultiTask):
    def __init__(self, dataset_names, image_steps=6, out_dropout=0.2, **model_kwargs):
        torch_S3D.__init__(self, **model_kwargs)
        MultiTask.__init__(self, dataset_names, image_steps)

        self.classifiers = self.generate_classifiers(1024)
        self.dropout = nn.Dropout(p=out_dropout)

    def forward(self, x):
        f = self.features(x)


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
])

def transform_video(video):
    frames = video.to_numpy() / 255.
    frames = torch.Tensor(frames)
    frames = frames.permute(0, 3, 1, 2)
    frames = img_transform(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier):

    # template for the identifier: {args.model}-{'.'.join(args.datasets)}-{args.lr}-{args.batch_size}
    model, dataset, lr, batch_size = identifier.split('-')
    datasets = dataset.split('.')
    assert model == "S3D"

    model = S3D(datasets)

    root = '/home/ytang/workspace/data/acache/extra'
    model_pth = os.path.join(root, f'{identifier}.pth')

    # Load the model weights
    model.load_state_dict(torch.load(model_pth, map_location='cpu'))

    inferencer_kwargs = {
        "fps": 16,
        "num_frames": (6, np.inf),
        "layer_activation_format": 
        {
            **{f"features.{i}": "CTHW" for i in range(16)},
        },
    }
    process_output = None

    wrapper = PytorchWrapper(identifier, model, transform_video, 
                             process_output=process_output,
                             **inferencer_kwargs)
    
    return wrapper