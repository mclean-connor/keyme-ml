import numpy as np
import cv2

from models.callbacks import BaseEvalCallback


class EvalCallback(BaseEvalCallback):
    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, num_samples=5
    ):
        super().__init__(
            validation_data, data_table_columns, pred_table_columns, num_samples
        )

    def _process_pred(self, pred):
        # Ensure pred is in the [0, 1] range before scaling
        pred = np.clip(pred, 0, 1)
        pred = np.tile(pred, [1, 1, 3])
        pred = cv2.cvtColor(pred.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0 # type: ignore

        return pred
