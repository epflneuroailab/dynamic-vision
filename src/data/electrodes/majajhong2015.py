from brainscore_vision import load_dataset as bs_load_dataset


def load_dataset(identifier):
    region = identifier.split('-')[-1]
    dataset = bs_load_dataset("MajajHong2015.public")
    dataset = dataset.sel(region=region)  #.reset_index('presentation').rename(presentation='stimulus_id')
    return dataset