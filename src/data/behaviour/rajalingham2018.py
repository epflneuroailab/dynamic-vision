import brainscore_vision


def _correct_wtf_posixpath(data):
    data.stimulus_paths = {k: str(v) for k, v in data.stimulus_paths.items()}
    return data

def load_dataset():
    assembly = brainscore_vision.load_dataset(f'Rajalingham2018.public')
    assembly['correct'] = assembly['choice'] == assembly['sample_obj']
    _correct_wtf_posixpath(assembly.stimulus_set)
    return assembly

def load_stimulus_set():
    data = brainscore_vision.load_stimulus_set('objectome.public')
    _correct_wtf_posixpath(data)
    return data