import os
import tensorflow as tf
import wandb
from wandb.keras import WandbEvalCallback, WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
import cv2


class EvalCallback(WandbEvalCallback):
    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, num_samples=5
    ):
        super().__init__(data_table_columns, pred_table_columns)
        # Unbatch and take the desired number of samples from the validation dataset
        self.validation_data = validation_data.unbatch().take(num_samples)

        batch = []
        for input, _ in self.validation_data.as_numpy_iterator():
            # Store the input images in a batch
            batch.append(input)

        self.x_batch = np.array(batch)

    def add_ground_truth(self, logs=None):
        # Iterate over the samples in the validation dataset
        for idx, (input, mask) in enumerate(self.validation_data.as_numpy_iterator()):
            # Add data to the wandb data table
            # Each sample is indexed by idx, and we log the input image and its corresponding mask
            self.data_table.add_data(idx, wandb.Image(input), wandb.Image(mask))

    def add_model_predictions(self, epoch, logs=None):
        # Iterate over the validation data
        # Predict the entire batch of images
        preds = self.model.predict(self.x_batch)[0]
        for idx, pred in enumerate(preds):
            pred = np.asarray(pred)

            # Ensure pred is in the [0, 1] range before scaling
            pred = np.clip(pred, 0, 1)
            pred = np.tile(pred, [1, 1, 3])
            pred = cv2.cvtColor(pred.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0  # type: ignore

            # Retrieve the data table reference and its indices
            data_table_ref = self.data_table_ref

            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],  # type: ignore
                data_table_ref.data[idx][1],  # type: ignore
                data_table_ref.data[idx][2],  # type: ignore
                wandb.Image(pred),
            )

    def on_train_batch_end(self, batch, logs=None):
        if batch % 250 == 0:
            # Predict the entire batch of images
            preds = self.model.predict(self.x_batch)[0]

            # Process predictions and concatenate images
            combined = None
            for idx, pred in enumerate(preds):
                pred = np.asarray(pred)

                # Ensure pred is in the [0, 1] range before scaling
                pred = np.clip(pred, 0, 1)
                pred = np.tile(pred, [1, 1, 3])
                pred = cv2.cvtColor(pred.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0  # type: ignore

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
                combined = pred if combined is None else cv2.hconcat([combined, pred])

            # Log combined image to wandb
            wandb.log(
                {f"Combined Training Predictions Over Time": wandb.Image(combined)}
            )


class MetricsCallback(WandbMetricsLogger):
    def __init__(self):
        super().__init__(
            "epoch",
        )


class ModelCheckpoint(WandbModelCheckpoint):
    def __init__(self):
        base_dir = os.path.join(wandb.run.dir, "weights")  # type: ignore
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(wandb.run.dir, "weights", "best-model.h5")  # type: ignore
        super().__init__(
            file_path,
            monitor="val_loss",  # save the model with the lowest validation loss
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )


def get_callbacks(validation_data):
    return [
        MetricsCallback(),
        ModelCheckpoint(),
        EvalCallback(
            validation_data,
            data_table_columns=["Idx", "Input", "Ground Truth Mask"],
            pred_table_columns=[
                "Epoch",
                "Idx",
                "Input",
                "Ground Truth Mask",
                "Predicted Mask",
            ],
        ),
    ]
