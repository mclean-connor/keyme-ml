from typing import List
import numpy as np

from models.classification.BittingFlatMetalDetectorModel.subclass import (
    BittingFlatMetalDetectorSubclass,
)
from models.classification.BittingFlatMetalDetectorModel import FlatDetectorConfig
from models.base_model import BaseKeyMeModel


class BittingFlatMetalDetector(BaseKeyMeModel):
    def __init__(self):
        self.config: FlatDetectorConfig
        super().__init__("BittingFlatMetalDetectorModel", config=FlatDetectorConfig())
        self.model = BittingFlatMetalDetectorSubclass(self.config.y_to_label)
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

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
