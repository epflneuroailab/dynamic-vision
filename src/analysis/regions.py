from neuroparc import atlas as natlas
from neuroparc.extra import glasser
import numpy as np
from .glasser import cortical_divisions, abbreviation_map


EARLY = ["V1", "V2", "V3"]

STREAMS = {
    "Early": EARLY,
    "Ventral": [
        "V4",
        "V8",
        "PIT",
        "FFC",
        "PH",
        "VVC",
        "VMV3",
        "VMV2",
        "VMV1",
        "PHA3",
        "PHA2",
        "PHA1",
        "TE1p",
        "TE2p",
        "TE1m",
        "TE2a",
        "TF"
    ],
    "Dorsal": [
        *EARLY,
        "V3A",
        "V3B",
        "V3CD",
        "V6",
        "V6A",
        "V7",
        "IPS1",

        "LO1", 
        "LO2", 
        "LO3", 
        "V4t",
        "MT", 
        "MST", 
        "FST", 
        "PH", 
        
        "IP0",
        "IP1",
        "IP2",
        "MIP",
        "VIP",
        "LIPd",
        "LIPv",
        "AIP",
        "7Pm",
        "7Pl",
        "7PC",
        "7Am",
        "7AL",
    ],
    "Dorso-dorsal": [
        "V3A",
        "V6",
        "V6A",
        "V7",
        "IPS1",
        "IP0",
        "IP1",
        # "IP2",

        "MIP",
        "VIP",
        "LIPd",
        "LIPv",
        "AIP",
        "7Pm",
        "7Pl",
        "7PC",
        "7Am",
        "7AL",
    ],
    "Ventro-dorsal": [
        "LO1", 
        "LO2", 
        "LO3", 
        "V4t",
        "MT", 
        "MST", 
        "FST", 
        "PH",
    ]
}
STREAMS = {k:[abbreviation_map[e] for e in v] for k,v in STREAMS.items()}


class MAP:
    regions = glasser.cortical_divisions
    atlas = natlas.Atlas("Glasser")
    label_surface = atlas.label_surface("fsaverage5")
    rev_label_name_map = {k.lower(): v for k, v in atlas.rev_label_name_map.items()}
    destrieux_atlas = natlas.Atlas("Destrieux")
    destrieux_label_surface = destrieux_atlas.label_surface("fsaverage5")
    destrieux_rev_label_name_map = {k.lower(): v for k, v in destrieux_atlas.rev_label_name_map.items()}

    def get_region_labels(region):
        if isinstance(region, str):
            region = [region]
        labels = []
        for r in region:
            labels.append(MAP.rev_label_name_map[r.lower()])
        return labels

    def get_nsel(region):
        region_labels = MAP.get_region_labels(region)
        return np.isin(MAP.label_surface, region_labels)

MAP.regions.update(STREAMS)

def get_region_voxels(region):
    region = MAP.regions[region] if not isinstance(region, list) and region in MAP.regions else region
    nsel = MAP.get_nsel(region)
    return nsel