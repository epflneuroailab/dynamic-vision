import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from scripts.utils import *
from src.analysis.fdr import false_discovery_control


MODEL = "VJEPA-Temporal"
# MODEL = "convnext_large_imagenet_full_seed-0"

FIGURE_DIR = os.path.join(FIGURE_CACHE, 'f2c')
os.makedirs(FIGURE_DIR, exist_ok=True)

this_dir = os.path.dirname(os.path.realpath(__file__))
args = get_args()
fmri, models = collect(args, types=["fmri"], models=[MODEL])
fmri_scores = fmri[MODEL].mean(0)
fmri_stds = fmri[MODEL].std(0)
ceiling = get_ceiling(args)
fmri_scores = select_and_ceil(fmri_scores, ceiling)

# stats
ts = fmri_scores / fmri_stds
pvals = 2 * norm.sf(np.abs(ts))
pvals = np.clip(pvals, 1e-20, 1 - 1e-20)
valid = ~np.isnan(fmri_scores)
pvals[valid] = false_discovery_control(pvals[valid])
fmri_scores[pvals > 0.05] = np.nan

for hemi, view in plot_volume_nilearn(fmri_scores, cmap='inferno', colorbar=True, vmax=0.8, vmin=0):
    plt.savefig(os.path.join(FIGURE_DIR, f'{hemi}_{view}.png'), dpi=300)