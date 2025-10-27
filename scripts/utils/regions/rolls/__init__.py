# bidirection arrows are simplified to single arrows (the larger arrow is kept)

STREAMS = [
    # blue: ventrolateral
    ("V1", "V2"),
    ("V2", "V3"),
    ("V3", "V4"),
    ("V4", "PIT"),
    ("V4", "V8"),
    ("V8", "FFC"),
    ("PIT", "PH"),
    ("PIT", "FFC"),
    ("FFC", "VVC"),
    ("FFC", "TE2p"),
    ("FFC", "TF"),
    ("FFC", "MT"),
    ("FFC", "TPOJ1"),
    ("FFC", "PH"),
    ("PH", "TE2p"),
    ("PH", "PHT"),
    ("PHT", "TE1p"),
    ("PHT", "TE2p"),
    ("TE1p", "TE1m"),
    ("TE1p", "TE2a"),
    ("TE1p", "PGs"),
    ("TE2p", "TF"),
    ("TE2p", "13l"),
    ("TE2p", "47m"),
    ("TF", "TGd"),
    ("TF", "EC"),
    ("TE2a", "TGv"),
    ("TE2a", "TE1m"),
    ("TE2a", "TE1a"),
    ("TE1a", "TGd"),
    ("TE1a", "PGi"),
    ("TE1m", "TE1a"),
    ("TE1m", "STSvp"),

    # green: ventromedial
    ("V2", "POS1"),
    ("V2", "DVT"),
    ("DVT", "VMV1"),
    ("DVT", "VMV2"),
    ("POS1", "VMV1"),
    ("POS1", "VMV2"),
    ("VMV1", "VMV2"),
    ("VMV1", "MST"),
    ("VMV1", "PGp"),
    ("VMV2", "PHA3"),
    ("VMV3", "PHA3"),
    ("VMV3", "VVC"),
    ("VVC", "VMV3"),
    ("V8", "VMV3"),
    ("V4", "VMV3"),
    ("PHA3", "PeEc"),

    # lateral
    ("PGi", "STSva"),
    ("PGi", "STSvp"),
    ("STSvp", "45"),
    ("STSvp", "PGi"),
    ("STSvp", "47l"),
    ("STSvp", "4SFL"),
    ("STSvp", "9m"),
    ("STSvp", "10pp"),
    ("STSvp", "47s"),
    ("STSvp", "TGv"),
    ("STSvp", "31pd"),
    ("TGv", "STSvp"),
    ("STSva", "TE1a"),
    ("TGd", "STSva"),
    ("STSva", "TGd"),
    ("STSva", "PGi"),
    ("10v", "STSva"),

    # dorsol
    ("V3", "LO3"),
    ("V3", "V3B"),
    ("LO3", "MT"),
    ("MT", "MST"),
    ("MST", "FST"),
    ("FST", "PH"),
    ("FST", "LIPv"),
    ("FST", "LIPd"),
    ("PH", "FST"),
    ("PH", "AIP"),
    ("V3B", "V3A"),
    ("V3A", "V7"),
    ("V7", "V6A"),
    ("V6A", "LIPd"),
    ("LIPd", "PGp"),
    ("PGp", "PHA2"),
]


ROIS = [
    "V1", "V2", "V3", "V4", "PIT", "V8", "FFC", "PH", "PHT", "TE1p", "TE2p", "TF", "MT", "TPOJ1", "VVC",
    "STSvp", "STSva", "LO3", "V3B", "MT", "MST", "FST", "LIPv", "LIPd", "AIP", "V3A", "V7", "V6A", "PGp", "PHA2", "PHA3",
    "VMV1", "VMV2", "VMV3", "POS1", "DVT", "PGi",
]


import src.config
from .anneal import get_probabilistic_hierarchy
from src.store import pickle_store

if pickle_store.exists("rolls"):
    regions, heatmap, hier = pickle_store.load("rolls")
else:
    regions, heatmap, hier = get_probabilistic_hierarchy(STREAMS, num_runs=100)
    for r, h in zip(regions, hier):
        print(r, h)
    pickle_store.store((regions, heatmap, hier), "rolls")

mask = [r in ROIS for r in regions]
heatmap = heatmap[mask][:, mask]
regions = [r for r in regions if r in ROIS]
hier = hier[mask]