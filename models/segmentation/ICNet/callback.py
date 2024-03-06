from typing import Dict
import numpy as np
from typing import Any, Dict, Optional
import wandb
import cv2

from models.callbacks import BaseEvalCallback, BaseMetricsCallback


class EvalCallback(BaseEvalCallback):
    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, num_samples=8
    ):
        super().__init__(
            validation_data, data_table_columns, pred_table_columns, num_samples
        )

    def _process_pred(self, pred):
        # prediction is shape (H, W, 2) where the last dimension is the class probabilities
        # Use argmax to get class indices
        pred = np.argmax(pred, axis=-1)

        # Convert binary class indices (0s and 1s) to 0s and 255s for proper black and white representation
        pred = pred * 255  # This will convert 1s to 255, making them white

        # Tile the array to get an RGB image
        pred = np.tile(
            pred[:, :, None], [1, 1, 3]
        )  # Add a dimension before tiling to avoid broadcasting issues

        return pred.astype("uint8")  # Ensure the data type is uint8 for an image

    def on_train_batch_end(self, batch, logs=None):
        if batch % 200 == 0:
            # Predict the entire batch of images
            preds = self.model.predict(self.x_batch)[0]

            # Process predictions and concatenate images
            combined = None
            for idx, pred in enumerate(preds):
                pred = np.asarray(pred)

                output = self._process_pred(pred)

                # Add index text to the image
                cv2.putText(
                    pred,
                    str(idx),
                    (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Concatenate horizontally
                combined = (
                    output if combined is None else cv2.hconcat([combined, output])
                )

            # Log combined image to wandb
            wandb.log(
                {f"Combined Training Predictions Over Time": wandb.Image(combined)}
            )


class MetricsCallback(BaseMetricsCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        logs = dict() if logs is None else {f"epoch/{k}": v for k, v in logs.items()}

        logs["epoch/epoch"] = epoch

        wandb.log(logs)
