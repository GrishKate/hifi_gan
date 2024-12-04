import torch
import torch.nn.functional as F

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}
    audio_len = 0
    mel_len = 0
    pad_value = -11.5129251
    load_audio = False
    if 'audio' in dataset_items[0].keys() and dataset_items[0]['audio'] is not None:
        load_audio = True
    for elem in dataset_items:
        if load_audio:
            audio_len = max(audio_len, elem['audio'].shape[0])
        mel_len = max(mel_len, elem['mel'].shape[1])
    if load_audio:
        result_batch['real'] = torch.vstack([F.pad(elem["audio"], (0, audio_len - elem["audio"].shape[0])) for elem in dataset_items]).unsqueeze(1)
    result_batch['real_mel'] = torch.stack([F.pad(elem["mel"], (0, mel_len - elem["mel"].shape[1]), value=pad_value) for elem in dataset_items])
    if 'id' in dataset_items[0].keys():
        result_batch['id'] = [elem['id'] for elem in dataset_items]
    if 'trans' in dataset_items[0].keys():
        result_batch['text'] = [elem['trans'] for elem in dataset_items]
    return result_batch
