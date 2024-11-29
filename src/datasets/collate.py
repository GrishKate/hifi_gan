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
    for elem in dataset_items:
        audio_len = max(audio_len, elem['audio'].shape[0])
        mel_len = max(mel_len, elem['mel'].shape[1])
    result_batch['real'] = torch.vstack([F.pad(elem["audio"], (0, audio_len - elem["audio"].shape[0])) for elem in dataset_items]).unsqueeze(1)
    result_batch['real_mel'] = torch.stack([F.pad(elem["mel"], (0, mel_len - elem["mel"].shape[1]), value=pad_value) for elem in dataset_items])
    return result_batch