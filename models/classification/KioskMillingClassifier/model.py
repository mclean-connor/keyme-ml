from typing import List
import numpy as np
from models.classification.KioskMillingClassifier.subclass import (
    KioskMillingClassifierSubclass,
)
from models.classification.KioskMillingClassifier import MillingConfig
from models.base_model import BaseKeyMeModel


class KioskMillingClassifier(BaseKeyMeModel):
    def __init__(self):
        self.config: MillingConfig
        super().__init__("BittingFlatMetalDetectorModel", MillingConfig())
        self.model = KioskMillingClassifierSubclass(self.config.y_to_label)

    def _train(self, imgs: List[str]) -> None:
        """
        Train the model on the given images
        """
        raise NotImplementedError(
            "Train method not implemented. Please override the _train method."
        )

    def _evaluate(self, imgs: List[str]) -> None:
        """
        Evaluate the model on the given images
        """
        raise NotImplementedError(
            "Evaluate method not implemented. Please override the _evaluate method."
        )

    def _predict(self, image_path: str) -> np.ndarray:
        """
        Predict the mask for the given image
        """
        raise NotImplementedError(
            "Predict method not implemented. Please override the _predict method."
        )
