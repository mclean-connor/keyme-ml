import tensorflow as tf
import wandb
import cv2
import numpy as np
import os
from PIL import Image
import pathlib
from typing import Optional

# project imports
from models.base_model import BaseKeyMeModel
from models.segmentation.ICNet.subclass import icnet_builder, ICNetLoss
from models.segmentation.utils import (
    format_input,
    load_training_images,
)
from models.segmentation.ICNet import ICNetConfig
from models.callbacks import get_callbacks
from models.segmentation.ICNet.callback import EvalCallback, MetricsCallback


class ICNet(BaseKeyMeModel):
    def __init__(
        self,
        name: str = "ICNet",
        config: Optional[ICNetConfig] = None,
        *args,
        **kwargs,
    ):
        if config is None:
            config = ICNetConfig()

        super().__init__(name, config, *args, **kwargs)
        num_classes = 2
        net = icnet_builder(
            pretrained_backbone=False,
            classes=num_classes,
            aux=True,
            in_size=(self.config.default_in_shape[0], self.config.default_in_shape[1]),
            fixed_size=False,
        )
        input = tf.keras.Input(shape=self.config.default_in_shape, name="input")
        output = net(input)

        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.loss = ICNetLoss(num_classes)
        self.optimizer = tf.keras.optimizers.Adam()

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=["accuracy"],
        )
        self.model.summary()

    def _predict(self, img_path: str) -> np.ndarray:
        image = Image.open(img_path).convert("RGB")  # type: ignore

        input_image = image
        if image.size != self.config.default_in_shape:
            input_image = image.resize(
                (self.config.default_in_shape[0], self.config.default_in_shape[1]),
                Image.BICUBIC,
            )

        input_tensor = format_input(input_image, self.config)
        fused_mask_tensor = self.model(input_tensor, Image.BICUBIC)[0][0]  # type: ignore
        output_mask = np.asarray(fused_mask_tensor)

        output_mask = cv2.resize(output_mask, dsize=image.size)

        output_mask = np.tile(np.expand_dims(output_mask, axis=2), [1, 1, 3])
        output_image = cv2.cvtColor(output_mask.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0  # type: ignore

        return output_image

    def _train(self, train_imgs, eval_imgs):
        if self.config.resume:
            # load weights based on most recent training
            path = max(
                pathlib.Path(self.weights_dir).glob("*.h5"), key=os.path.getctime
            )
            print("Loading weights from %s" % path)
            self.model.load_weights(path)

        # load the training data
        train_ds = tf.data.Dataset.from_tensor_slices(train_imgs)
        train_label_ds = train_ds.map(
            lambda img_path: load_training_images(
                img_path, self.config, expand_dims=False
            )
        )
        training_data = (
            train_label_ds.shuffle(buffer_size=600)
            .batch(self.config.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # load the validation data
        val_ds = tf.data.Dataset.from_tensor_slices(eval_imgs)
        val_label_ds = val_ds.map(
            lambda img_path: load_training_images(
                img_path, self.config, expand_dims=False
            )
        )
        val_data = val_label_ds.batch(self.config.batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        callbacks = get_callbacks(
            validation_data=val_data,
            eval_callback=EvalCallback,
            metrics_callback=MetricsCallback,
        )
        for callback in callbacks:
            callback.model = self.model

        for callback in callbacks:
            callback.on_train_begin()

        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch}/{self.config.epochs}")

            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, (x_batch_train, y_batch_train) in enumerate(training_data):
                print(
                    f"Step {step}/{len(train_imgs) // self.config.batch_size} \n ",
                    end="\r",
                )
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch_train, training=True)
                    loss_value = self.loss(y_batch_train, y_pred)

                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                for callback in callbacks:
                    callback.on_train_batch_end(batch=step, logs={"loss": loss_value})

            # Validation phase
            val_loss_total = 0
            val_steps = 0
            for x_batch_val, y_batch_val in val_data:
                val_pred = self.model(x_batch_val, training=False)
                val_loss = self.loss(y_batch_val, val_pred)
                val_loss_total += val_loss
                val_steps += 1

            val_loss_avg = val_loss_total / val_steps
            print(f"Validation loss for epoch {epoch}: {val_loss_avg}")

            # Log validation loss to WandB
            wandb.log({"val_loss": val_loss_avg}, step=epoch)

            # Epoch-end callbacks
            logs = {"val_loss": val_loss_avg}  # Add other validation metrics if needed
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)

        # Finalize training
        for callback in callbacks:
            callback.on_train_end()
