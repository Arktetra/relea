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
                {f"train/batch/{k}": v for k, v in trainer.losses.items()}
            )

    def after_epoch(self, trainer: "relea.Trainer"):
        log_dict = {}

        for k, m in trainer.metrics.all_metrics.items():
            log_dict[k] = m.compute()
    
        self.logger.log(log_dict)
        

    def after_train(self, trainer: "relea.Trainer"):
        self.logger.finish()