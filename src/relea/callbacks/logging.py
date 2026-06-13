from relea.callbacks import Callback

import relea

class LoggingCallback(Callback):
    order = 2

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def after_batch(self, trainer: "relea.Trainer"):
        if trainer.training:
            self.logger.log(
                {f"train/batch/{k}": trainer.model.__dict__[k] for k in self.keys}
            )

    def _log_metrics(self, trainer: "relea.Trainer"):
        log_dict = {}

        for k, m in trainer.metrics.all_metrics.items():
            log_dict[k] = m.compute()
    
        self.logger.log(log_dict)

    def after_eval(self, trainer: "relea.Trainer"):
        self._log_metrics(trainer)

    def after_epoch(self, trainer: "relea.Trainer"):
        self._log_metrics(trainer)

    def before_train(self, trainer: "relea.Trainer"):
        self.keys = [k for k in trainer.model.__dict__.keys() if "loss" in k]

    def after_train(self, trainer: "relea.Trainer"):
        self.logger.finish()
