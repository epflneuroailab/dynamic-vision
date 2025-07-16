import os
import cortex
import numpy as np
from neuroparc.atlas import Atlas
from matplotlib import pyplot as plt
from .task_color import map_rgb, map_dual, _map_dual

subject = "fsaverage"
# First we check if the fsaverage template is already in the pycortex filestore. If not,
# we download the template from the web and add it to the filestore.
if subject not in cortex.db.subjects:
    cortex.download_subject(subject)

cortex_args = {
    "with_curvature": True,
    "with_rois": False,
    "with_labels": True,
    "with_sulci": False,
    "with_colorbar": False,
}


def plot_single_factor(data_fs5, vmin=None, vmax=None, cmap=None, **kwargs):
    surface = Atlas(subject, data_fs5)
    data_fs7 = surface.label_surface("fsaverage7")
    vtx = cortex.Vertex(data_fs7, subject, vmin=vmin, vmax=vmax, cmap=cmap)
    cortex_args.update(kwargs)
    cortex.quickflat.make_figure(
        vtx,
        **cortex_args,
    )

def plot_dual(data_fs5, color1=None, color2=None, **kwargs):
    data_fs7 = Atlas(subject, data_fs5).label_surface("fsaverage7")
    if color1 is None or color2 is None:
        rgbs = map_dual(data_fs7)
    else:
        rgbs = _map_dual(data_fs7, color1, color2)

    red = cortex.Vertex(rgbs[...,0], subject, 'fullhead', vmax=1, vmin=0)
    green = cortex.Vertex(rgbs[...,1], subject, 'fullhead', vmax=1, vmin=0)
    blue = cortex.Vertex(rgbs[...,2], subject, 'fullhead', vmax=1, vmin=0)
    vtx = cortex.VertexRGB(red, green, blue, subject)

    cortex_args.update(kwargs)

    cortex.quickflat.make_figure(
        vtx,
        **cortex_args,
    )

def plot_factors(uni_r2s, full_r2, threshold=0.4):
    img = Atlas(subject, uni_r2s[0]).label_surface("fsaverage7")
    kin = Atlas(subject, uni_r2s[1]).label_surface("fsaverage7")
    afd = Atlas(subject, uni_r2s[2]).label_surface("fsaverage7")
    full = Atlas(subject, full_r2).label_surface("fsaverage7")

    scores = np.stack([img, kin, afd], axis=-1)
    nans = np.isnan(scores)
    scores[nans|(scores<0)] = 0
    scores = scores / np.nanmax(scores)
    colors = (map_rgb(scores) * 255).astype(np.uint8)

    red = cortex.Vertex(colors[...,0], subject, 'fullhead')
    green = cortex.Vertex(colors[...,1], subject, 'fullhead')
    blue = cortex.Vertex(colors[...,2], subject, 'fullhead')

    alpha = full
    nans = np.isnan(alpha)
    alpha = alpha >= threshold
    alpha[nans] = 0

    vtx = cortex.VertexRGB(red, green, blue, subject)
    vtx = vtx.blend_curvature(alpha)

    cortex.quickflat.make_figure(
        vtx,
        **cortex_args
    )


def plot_volume_nilearn(data, **kwargs):
    from nilearn import plotting
    from nilearn import image, datasets

    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    data = {
        "left": data[:10242],
        "right": data[10242:],
    }

    if "colorbar" not in kwargs:
        kwargs["colorbar"] = False
    if "darkness" not in kwargs:
        kwargs["darkness"] = 0.5

    for hemi in ['left', 'right']:
        for view in ['lateral', 'medial', 'dorsal', 'ventral']:
            plotting.plot_surf(
                surf_mesh=fsaverage[f'pial_{hemi}'],
                surf_map=data[hemi],
                hemi=hemi,
                view=view,
                bg_map=fsaverage[f'sulc_{hemi}'],
                **kwargs
            )
            yield hemi, view


def plot_surface_nilearn(data, **kwargs):
    from nilearn import plotting
    from nilearn import image, datasets

    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

    if "colorbar" not in kwargs:
        kwargs["colorbar"] = False
    if "darkness" not in kwargs:
        kwargs["darkness"] = 0.5

    data = {
        "left": data[:10242],
        "right": data[10242:],
    }

    # plot flat map
    for hemi in ['left', 'right']:
        plotting.plot_surf(
            surf_mesh=fsaverage[f'infl_{hemi}'],
            surf_map=data[hemi],
            hemi=hemi,
            bg_map=fsaverage[f'sulc_{hemi}'],
            **kwargs
        )
        yield hemi