# Dynamic‑Vision

**Dynamic‑Vision** is a benchmarking framework for testing computer
vision models on *dynamic* stimuli and
quantifying how well they align with neural and behavioural data.  It
combines model activation extraction, simple decoders and evaluation
metrics into a coherent pipeline.  At a high level Dynamic‑Vision
loads a model, computes its spatiotemporal activations on a given
dataset and uses linear decoders to predict behavioural choices,
fMRI signals or standard computer‑vision tasks.

The code builds on the
[BrainScore‑Vision](https://github.com/brain-score/vision) tools for
loading models and brain datasets.  Static image architectures are
optionally converted into temporal models, and a handful of decoders
(ridge regression or logistic regression) are used to report
alignment scores.
The complete dataset loading process is currently in progress and will be updated shortly.

## Structure

The top‑level `src` package contains all implementation details and
four scoring scripts.  Key components include:

* **`analysis`** – statistical helpers for permutation tests,
  cross‑validation and false discovery control.
* **`data`** – dataset loaders and definitions.  `src/data/__init__.py`
  enumerates fMRI datasets, behavioural experiments, electrode
  recordings and standard computer‑vision tasks and maps each to an
  appropriate decoder.
* **`evaluate`** – decoders and evaluation functions for
  behaviour, fMRI, electrodes and classification tasks.
* **`models`** – model loaders and definitions.  `groups.py` lists a
  wide array of image and video models, from ResNets and EfficientNets
  to R3D, SlowFast and masked autoencoders.  The loader in
  `loading.py` converts static models to temporal ones and
  configures inference.
* **`utils.py`, `store.py` and `ceiling.py`** – utilities for caching
  results and computing noise ceilings.
* **Scoring scripts** – `score_behaviour.py`, `score_electrodes.py`,
  `score_fmri.py` and `score_task.py`.  Each accepts a model
  identifier plus one or more tasks/datasets and outputs alignment
  scores.

Running a script loads the requested model, computes activations,
fits a linear decoder and reports test/validation scores.  Results
and activations are cached on disk.

## Datasets and tasks

The `data` module defines several groups of inputs:

* **fMRI datasets** – the `DATASETS` list includes multiple
  recordings such as *savasegal2023*, *keles2024* and *mcmahon2023*.
* **Behavioural tasks** – experiments like *ilic2022–ucf5* and
  *rajalingham2018* are listed in `BEHAVIOURS` and come with built‑in
  metric.
* **Electrode recordings** – recordings from areas V1, V2, V4, IT and
  CRCNS datasets are enumerated in `ELECTRODES`.
* **Standard tasks** – classification or action‑recognition tasks
  (ImageNet‑100, Kinetics‑400, self‑motion and others) are defined in
  `TASKS` along with the appropriate decoder type.  A
  separate `STATIC` list identifies tasks that involve single images
  rather than video frames.

## Supported models

Dozens of models are available out‑of‑the‑box.  Static image networks
(e.g. ResNet, EfficientNet, DeiT and ConvNeXt variants)
can be transformed into temporal models.  Video‑centric architectures
include 3D CNNs and transformers such as R3D‑18, SlowFast, Video Swin
and TimeSformer, as well as masked autoencoders,
audio‑video networks and recurrent predictors.  See
`src/models/groups.py` for the full list.  The loader handles
downsampling, temporal conversion and random initialization.

## Setup

```bash
# install core dependencies
pip install numpy scipy pandas xarray scikit-learn torch torchvision
pip install h5py tables matplotlib opencv-python requests
pip install brainscore-core brainscore-vision brainio
```

Set `RESULTCACHING_HOME`, `BRAINIO_HOME`, `BRAINSCORE_HOME`, `TORCH_HOME`,
`HF_HOME` and `MMAP_HOME` to directories on your machine where the
code can store intermediate results and locate stimuli.  See
`src/config.py` for examples.

## Usage

Activate your environment, ensure the cache variables are set and run
one of the scoring scripts.  Each takes a `--model` argument plus
dataset/task names.  Examples:

* **Behaviour** – compare a model to human choices:
  ```bash
  python score_behaviour.py --model alexnet --tasks rajalingham2018
  ```
* **Electrodes** – fit the model to neuronal recordings:
  ```bash
  python score_electrodes.py --model resnet50_imagenet_full \
    --datasets freemanziemba2013-V4 crcns-pvc1
  ```
* **fMRI** – evaluate on one or more fMRI datasets (optionally convolving with an HRF):
  ```bash
  python score_fmri.py --model r3d_18 --datasets mcmahon2023-fmri keles2024-fmri
  ```
* **Tasks** – run standard vision benchmarks using logistic or ridge decoding:
  ```bash
  python score_task.py --model resnet50_imagenet_full --tasks imagenet2012 selfmotion
  ```

## Caching

Intermediate activations and scores are cached on disk via
`resultcaching` and `pickle_store`. Caches live under the
`RESULTCACHING_HOME` directory and make repeated evaluations faster.
Use `--rerun-activation` and related flags when you want to force
recomputation.
