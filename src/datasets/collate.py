import torch


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
    result_batch['real'] = torch.vstack([elem["audio"] for elem in dataset_items])
    result_batch['real_mel'] = torch.vstack([elem["mel"] for elem in dataset_items])
    return result_batch
