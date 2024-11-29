import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class MelMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

    def __call__(self, real_mel, fake_mel, **kwargs):
        """
        Metric calculation logic.

        Args:
            real_mel and fake_mel
        Returns:
            metric (float): calculated metric.
        """
        with torch.no_grad():
            loss = F.l1_loss(real_mel, fake_mel).item()
        return loss