"""
Custom metrics.
"""

from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
    MulticlassF1Score
)

from relea.callbacks.metrics import MetricsCallback

class ClassificationMetricsCallback(MetricsCallback):
    """
    Metrics callback for image generative models.

    Expects the Module wrapping the model to have a sample method for 
    sampling images.
    """
    def __init__(
        self, 
        num_classes: int = 10,
        verbose: bool = False, 
    ):
        self.metrics = {
            "acc": MulticlassAccuracy(average="macro", num_classes=num_classes),
            "precision": MulticlassPrecision(average="macro", num_classes=num_classes),
            "recall": MulticlassRecall(average="macro", num_classes=num_classes),
            "f1-score": MulticlassF1Score(average="macro", num_classes=num_classes)
        }
        super().__init__(val=True, verbose=verbose, **self.metrics)

    @staticmethod
    def from_config(
        cfg: dict
    ) -> "ClassificationMetricsCallback":
        num_classes = cfg["num_classes"]
        verbose = cfg["verbose"]
        return ClassificationMetricsCallback(num_classes=num_classes, verbose=verbose)
