from brainscore_vision import load_dataset as bs_load_dataset


def load_dataset(identifier):
    region = identifier.split('-')[-1]
    assembly = bs_load_dataset("FreemanZiemba2013.public")
    assembly = assembly.sel(region=region)  #.reset_index('presentation').rename(presentation='stimulus_id')
    assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    time_window = (50, 200)
    assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(*time_window)])
    assembly = assembly.mean(dim='time_bin', keep_attrs=True)
    assembly = assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
    assembly['time_bin_start'], assembly['time_bin_end'] = [time_window[0]], [time_window[1]]
    assembly = assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
    return assembly
