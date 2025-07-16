import colorsys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def hex_to_rgb(hex):
    return np.array(tuple(int(hex[i:i+2], 16) for i in (1, 3, 5))) / 255

def map_rgb(vals, rot=0.5):
    # vals [..., 3]

    assert vals.shape[-1] == 3
    assert (vals >= 0).all()

    # project to triangular plane
    vals = vals / (np.sum(vals, axis=-1, keepdims=True)+1e-6)
    vals = vals - np.array([1, 1, 1]) / 3

    # to triangular coordinates
    tri_1 = np.array([-1, 1, 0])
    tri_2 = np.array([-1, -1, 2])
    tri_1 = tri_1 / np.linalg.norm(tri_1)
    tri_2 = tri_2 / np.linalg.norm(tri_2)
    proj = np.array([tri_1, tri_2])
    vals = np.dot(vals, proj.T)
    
    # to polar coordinates
    r = np.linalg.norm(vals, axis=-1)
    theta = np.arctan2(vals[..., 1], vals[..., 0])

    # triangulation correction
    ang = theta / (2*np.pi) * 360
    ang = (30 + ang) % 120
    ang = abs(ang - 60)
    length = .5 / np.cos(np.radians(ang))
    r /= length

    # rotation
    theta += np.pi * rot

    # map to color
    hue = (theta + np.pi) / (2 * np.pi)
    saturation = r

    # convert to rgb
    hsv = np.stack([hue, saturation, np.ones_like(hue)], axis=-1)
    to_rgb = lambda x: colorsys.hsv_to_rgb(x[0], x[1], x[2])
    rgb = np.apply_along_axis(to_rgb, -1, hsv)

    return rgb


def map_rgb_(vals):
    # vals [..., 3]

    assert vals.shape[-1] == 3
    assert (vals >= 0).all()

    vals /= vals.max()
    return vals


def get_corners():
    return map_rgb(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]))


def map_dual(vals):
    from matplotlib import colormaps

    vals = 1 - vals
    cmap = colormaps.get_cmap("Spectral")
    nans = np.isnan(vals)
    vals[nans] = 0

    rgb = cmap(vals)
    rgb[nans] = np.nan
    return rgb

def _map_dual(vals, color1, color2):
    # Define two hex colors

    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    hsv1 = colorsys.rgb_to_hsv(*rgb1)
    hsv2 = colorsys.rgb_to_hsv(*rgb2)

    nans = np.isnan(vals)
    vals[nans] = 0

    # Interpolate between the two colors
    hsv = np.array([hsv1[i] * (1-vals) + hsv2[i] * (vals) for i in range(3)]).T
    rgb = np.array([colorsys.hsv_to_rgb(*h) for h in hsv])

    rgb[nans] = np.nan
    return rgb


def get_colorbar():
    vals = np.linspace(0, 1, 256)
    rgbs = map_dual(vals)
    return rgbs