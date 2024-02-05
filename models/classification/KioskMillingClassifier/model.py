from typing import Any
from numpy import ndarray
from models.classification.KioskMillingClassifier.subclass import (
    KioskMillingClassifierSubclass,
)
from models.classification.KioskMillingClassifier import config
from models.base_model import BaseKeyMeModel


class KioskMillingClassifier(BaseKeyMeModel):
    def __init__(self):
        super().__init__("BittingFlatMetalDetectorModel", config)
        self.model = KioskMillingClassifierSubclass(config.y_to_label)
