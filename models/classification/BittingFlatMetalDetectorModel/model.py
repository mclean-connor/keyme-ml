from models.classification.BittingFlatMetalDetectorModel.subclass import (
    BittingFlatMetalDetectorSubclass,
)
from models.classification.BittingFlatMetalDetectorModel import config
from models.base_model import BaseKeyMeModel


class BittingFlatMetalDetector(BaseKeyMeModel):
    def __init__(self):
        super().__init__("BittingFlatMetalDetectorModel", config)
        self.config = config

        self.model = BittingFlatMetalDetectorSubclass(config.y_to_label)
        self.model.compile(
            optimizer=config.optimizer,
            loss=config.loss,
            metrics=config.metrics,
        )
