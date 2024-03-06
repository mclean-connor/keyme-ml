import tensorflow as tf
from tensorflow import keras  # type: ignore
import cv2
import numpy as np
import os
from PIL import Image
import pathlib
from typing import Optional

# project imports
from models.base_model import BaseKeyMeModel
from models.segmentation.U2Net.subclass import U2NET, U2NETP, bce_loss, average_iou
from models.segmentation.utils import (
    format_input,
    load_training_images,
)
from models.segmentation.U2Net import U2NetConfig
from models.segmentation.U2Net.callback import EvalCallback
from models.callbacks import get_callbacks


class U2Net(BaseKeyMeModel):
    def __init__(
        self,
        name: str = "U2Net",
        tiny_model: bool = False,
        config: Optional[U2NetConfig] = None,
        *args,
        **kwargs,
    ):
        if config is None:
            config = U2NetConfig()

        super().__init__(name, config, *args, **kwargs)
        inputs = keras.Input(shape=self.config.default_in_shape)
        net = U2NET() if not tiny_model else U2NETP()
        out = net(inputs)

        # Overwrite the default optimizer
        adam = keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
        )
        self.model = keras.Model(inputs=inputs, outputs=out, name="u2netmodel")
        self.model.compile(
            optimizer=adam,
            loss=bce_loss,
            metrics=[average_iou],
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
            lambda img_path: load_training_images(img_path, self.config)
        )
        training_data = (
            train_label_ds.shuffle(buffer_size=600)
            .batch(self.config.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        # load the validation data
        val_ds = tf.data.Dataset.from_tensor_slices(eval_imgs)
        val_label_ds = val_ds.map(
            lambda img_path: load_training_images(img_path, self.config)
        )
        val_data = val_label_ds.batch(self.config.batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        self.model.fit(
            training_data,
            epochs=self.config.epochs,
            callbacks=get_callbacks(
                validation_data=val_data, eval_callback=EvalCallback
            ),
            validation_data=val_data,
        )
