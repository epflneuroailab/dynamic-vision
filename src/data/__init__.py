from ..store import pickle_store

data_store = pickle_store.add_node("data")

DATASETS = [
    "fake-fmri",
    "savasegal2023-fmri-defeat",
    "savasegal2023-fmri-growth",
    "savasegal2023-fmri-iteration",
    "savasegal2023-fmri-lemonade",
    "keles2024-fmri",
    "berezutskaya2021-fmri",
    "mcmahon2023-fmri",
    "lahner2024-fmri",
]

TASKS = {
    "imagenet2012": "logistic_regression",
    "afd2022": "logistic_regression",
    "kinetics400": "logistic_regression",
    "kinetics400-static": "logistic_regression",
    "ding2012": "logistic_regression",
    "smthsmthv2": "logistic_regression",
    "vggface2": "logistic_regression",
    "hdm05": "logistic_regression",
    "selfmotion": "ridgecv",
    "mcmahon2023-social": "ridgecv",
    "majajhong2015-pose": "ridgecv",
    "cueconflict": "logistic_regression",
    "mechanical_tools": "logistic_regression",
}

BEHAVIOURS = {
    "ilic2022-ucf5": "O2",
    "ilic2022-afd5": "O2",
    "rajalingham2018": "I2n",
    "han2024-RGB": "O2",
    "han2024-RGB-4F": "O2",
    "han2024-RGB-S": "O2",
    "han2024-J-6P": "O2",
    "han2024-J-6P-4F": "O2",
    "han2024-J-6P-S": "O2",
}

ELECTRODES = [
    "freemanziemba2013-V1",
    "freemanziemba2013-V2",
    "majajhong2015-V4",
    "majajhong2015-IT",
    "crcns-pvc1",
    "crcns-mt2",
    "oleo-hacs250",
]

STATIC = [
    "imagenet2012",
    "kinetics400-static",
    "rajalingham2018",
    "rajalingham2018-fitting_stimuli",
    "vggface2",
    "majajhong2015-pose",
    "cueconflict",
    "mechanical_tools",
]