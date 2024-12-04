import os
from pathlib import Path
import torch
import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir=None, resynthesize=False, audio_dir=None, *args, **kwargs):
        self.resynthesize = resynthesize
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "dla_dataset"
        else:
            data_dir = Path(data_dir).absolute()
        if audio_dir is None:
            audio_dir = data_dir / 'trancriptions'
        else:
            audio_dir = Path(audio_dir).absolute()
        self._data_dir = data_dir
        self.audio_dir = audio_dir
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / "transcriptions"
        index = []
        if index_path.exists():
            for trans in os.listdir(str(index_path)):
                if trans.endswith((".txt")):
                    id = trans[:-4]
                    with open(os.path.join(index_path, trans), 'r') as f:
                        text = f.read()
                    d = {'id': id, 'trans': text, 'path': str(self.audio_dir) + '/' + id + '.wav'}
                    index.append(d)
                    if not self.resynthesize:
                        d['mel'] = torch.tensor(np.load(str(index_path / 'mel') + '/' + id + '.npy')).squeeze()
        else:
            raise RuntimeError
        return index

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        if self.resynthesize:
            audio = self.load_audio(data_dict['path']).squeeze()
            mel = self.make_mel(audio)
        else:
            mel = data_dict['mel']
            audio = self.load_audio(data_dict['path']).squeeze()

        instance_data = {'id': data_dict['id'], 'mel': mel,
                         'trans': data_dict['trans'], 'audio': audio}
        instance_data = self.preprocess_data(instance_data)
        return instance_data