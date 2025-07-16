import os
import pickle
import json
import shutil

from . import config

DEFAULT_ROOT = '/home/ytang/workspace/data/acache'

class PickleStore:
    def __init__(self, root=DEFAULT_ROOT):
        os.makedirs(root, exist_ok=True)
        self.root = root

    def add_node(self, name):
        new_root = f"{self.root}/{name}"
        return PickleStore(new_root)

    def _get_path(self, name):
        if name.endswith('.pth') or name.endswith('.pt') or name.endswith('.p'):
            return f"{self.root}/{name}"
        return f"{self.root}/{name}.p"

    def store(self, data, name):
        with open(self._get_path(name), 'wb') as f:
            pickle.dump(data, f)

    def load(self, name):
        with open(self._get_path(name), 'rb') as f:
            return pickle.load(f)

    def exists(self, name):
        return os.path.exists(self._get_path(name))


pickle_store = PickleStore()


from brainscore_vision.model_helpers.activations.temporal.utils import data_assembly_mmap
class ActivationStore:
    def __init__(self):
        self.root = os.environ.get('MMAP_HOME', None)
        assert self.root is not None, "Please set MMAP_HOME."

        self.table_path = os.path.join(self.root, 'table')
        os.makedirs(self.table_path, exist_ok=True)

    def search_path(self, name):
        path = os.path.join(self.table_path, name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return None

    def load(self, name):
        mmap_path = self.search_path(name)
        data = data_assembly_mmap.load(mmap_path)
        if data is None:
            self.drop(name)
            raise ValueError(f"Failed to load {mmap_path}")
        return data.to_assembly()

    def drop(self, name):
        record = os.path.join(self.table_path, name)
        if os.path.exists(record):
            os.remove(record)

    def clear(self, name):
        path = self.search_path(name)
        if path is not None and os.path.exists(path):
            shutil.rmtree(path)
        self.drop(name)

    def store(self, model_identifier, stimuli_identifier, identifier):
        mmap_path = os.path.join(self.root, stimuli_identifier, model_identifier)
        with open(os.path.join(self.table_path, identifier), 'w') as f:
            f.write(mmap_path)

    def exists(self, name):
        return self.search_path(name) is not None

activation_store = ActivationStore()