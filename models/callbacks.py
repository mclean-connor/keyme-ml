import os
import tensorflow as tf
import wandb
from wandb.keras import WandbEvalCallback, WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
import cv2


class BaseEvalCallback(WandbEvalCallback):
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

    def _process_pred(self, pred):
        # neet to be implemented in subclass, raise if not implamted
        raise NotImplementedError

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

            output = self._process_pred(pred)

            # Retrieve the data table reference and its indices
            data_table_ref = self.data_table_ref

            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],  # type: ignore
                data_table_ref.data[idx][1],  # type: ignore
                data_table_ref.data[idx][2],  # type: ignore
                wandb.Image(output),
            )

    def on_train_batch_end(self, batch, logs=None):
        if batch % 500 == 0:
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


class BaseMetricsCallback(WandbMetricsLogger):
    def __init__(self):
        super().__init__(
            "epoch",
        )


class BaseModelCheckpoint(WandbModelCheckpoint):
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


def get_callbacks(
    validation_data,
    eval_callback=BaseEvalCallback,
    metrics_callback=BaseMetricsCallback,
    checkpoint_callback=BaseModelCheckpoint,
):
    return [
        metrics_callback(),
        checkpoint_callback(),
        eval_callback(
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
