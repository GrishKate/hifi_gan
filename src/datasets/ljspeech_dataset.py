import json
import os
import shutil
from pathlib import Path
import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

PARTS = {
    "train": lambda x: x <= 10480,
    "test": lambda x: x > 10480
}


class LJSpeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "dla_dataset"
        else:
            data_dir = Path(data_dir).absolute()
        self._data_dir = data_dir
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / "metadata.csv"
        if index_path.exists():
            df = pd.read_csv('metadata.csv', sep='|', names=['id', 'trans', 'trans2'])
            df.drop(columns=['trans2'], inplace=True)
            df['audio_path'] = str(self._data_dir / "wavs") + '/' + df['id'] + '.wav'
            index = [df.iloc[i].to_dict() for i in range(len(df)) if PARTS[part](i)]
        else:
            raise RuntimeError
        return index
