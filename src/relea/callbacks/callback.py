from operator import attrgetter
from typing import Sequence

import relea
from relea.exceptions import (
    CancelBatchException,
    CancelEpochException,
    CancelTrainException,
)


class Callback:
    """
    Abstract base class for building new callbacks.
    """

    order = 0

    def before_batch(self, trainer: "relea.Trainer"):
        pass

    def after_batch(self, trainer: "relea.Trainer"):
        pass

    def before_epoch(self, trainer: "relea.Trainer"):
        pass

    def after_epoch(self, trainer: "relea.Trainer"):
        pass

    def before_train(self, trainer: "relea.Trainer"):
        pass

    def after_train(self, trainer: "relea.Trainer"):
        pass


class with_callbacks:
    def __init__(self, name):
        self.name = name

    def __call__(self, f):
        def _f(o: "relea.Trainer", *args, **kwargs):
            try:
                o.callback(f"before_{self.name}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.name}")
            except globals()[f"Cancel{self.name.title()}Exception"]:
                pass

        return _f


def run_callbacks(
    callbacks: Sequence[Callback], name: str, learner: "relea.Trainer"
) -> None:
    for callback in sorted(callbacks, key=attrgetter("order")):
        method = getattr(callback, name, None)
        if method is not None:
            method(learner)