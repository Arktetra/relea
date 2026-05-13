from relea.callbacks import Callback
from relea.exceptions import CancelEpochException

import relea


class NumBatchCallback(Callback):
    """
    Callback for training on only `num` number of batches on every epoch.

    Args:
    ---
        num (int): the number of batches to run on every epoch. Defaults to 1.
    """

    order = 0

    def __init__(self, num: int = 1):
        super().__init__()
        assert num > 0, "num must be a positive integer"
        self.num = num

    def after_batch(self, trainer: "relea.Trainer"):
        if (trainer.batch_idx + 1) == self.num:
            raise CancelEpochException()