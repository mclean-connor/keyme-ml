import tensorflow as tf
from tensorflow import keras  # type: ignore
import cv2
import numpy as np
import os
import wandb
import signal
from PIL import Image
import pathlib

# project imports
from models.base_model import BaseKeyMeModel
from models.segmentation.U2Net.subclass import U2NET, U2NETP, bce_loss
from models.segmentation.U2Net.utils import load_training_batch, format_input
from models.segmentation.U2Net import config
from models.utils import load_image_from_s3

# Overwrite the default optimizer
adam = keras.optimizers.Adam(
    learning_rate=config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
)
cp_callback = tf.keras.callbacks.ModelCheckpoint(  # type: ignore
    filepath=config.weights_file, save_weights_only=True, verbose=1
)


class U2Net(BaseKeyMeModel):
    def __init__(self, name: str = "U2Net", tiny_model: bool = False):
        super().__init__(name, config)
        inputs = keras.Input(shape=config.default_in_shape)
        net = U2NET() if not tiny_model else U2NETP()
        out = net(inputs)
        self.model = keras.Model(inputs=inputs, outputs=out, name="u2netmodel")
        self.model.compile(optimizer=adam, loss=bce_loss, metrics=None)

    def train(self, imgs):
        low_loss = float("inf")

        if config.resume:
            # load weights based on most recent training
            path = max(config.weight_dir.glob("*.h5"), key=os.path.getctime)
            print("Loading weights from %s" % path)
            self.model.load_weights(path)

        # helper function to save state of model
        def save_weights():
            print("Saving state of model to %s" % config.weights_file)
            self.model.save_weights(str(config.weights_file))
            wandb.save(str(config.weights_file))

        # # signal handler for early abortion to autosave model state
        # def autosave(sig, frame):
        #     print("Training aborted early... Saving weights.")
        #     save_weights()
        #     exit(0)

        # for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
        #     signal.signal(sig, autosave)

        # load the first image to get the shape
        in_image = load_image_from_s3(imgs[0]).convert("RGB")  # type: ignore
        # only keep the right half of the image... non mask part
        in_image = in_image.crop(
            (in_image.width // 2, 0, in_image.width, in_image.height)
        )

        # start training
        for e in range(config.epochs):
            try:
                feed, out = load_training_batch(
                    images=imgs, batch_size=config.batch_size
                )
                loss = self.model.train_on_batch(feed, out)
            except KeyboardInterrupt:
                save_weights()
                return
            except ValueError:
                print("ValueError: Skipping batch")
                continue

            if e % 10 == 0:
                print("[%05d] Loss: %.4f" % (e, loss))

            if loss < low_loss:  # type: ignore
                low_loss = loss
                print("[%05d] New Low Loss: %.4f" % (e, loss))
                # save the prediction
                fused_mask_tensor = self.model(feed[0:1], Image.BICUBIC)[0][0]  # type: ignore
                pred_mask = np.asarray(fused_mask_tensor)
                # resize the mask to the original image size
                pred_mask = cv2.resize(pred_mask, dsize=in_image.size)
                pred_mask = cv2.cvtColor(pred_mask.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0  # type: ignore

                wandb.log(
                    {"low_loss": low_loss, "epoch": e, "pred": wandb.Image(pred_mask)}
                )
                save_weights()

            # Log to Weights and Biases
            wandb.log({"loss": loss, "low_loss": low_loss, "epoch": e})

    def evaluate(self, imgs):
        def apply_mask(img, mask):
            return np.multiply(img, mask)

        input_images = []

        if not config.eval_dir.exists():
            config.eval_dir.mkdir()

        assert len(imgs) > 0, "No images given for evaluation"
        input_images.extend(imgs)

        if not config.output_dir.exists():
            config.output_dir.mkdir()

        if len(input_images) == 0:
            return

        # load weights based on most recent training
        weigths_path = max(config.run_dir.glob("*.h5"), key=os.path.getctime)
        self.model.load_weights(weigths_path)

        # evaluate each image
        for img in input_images:
            image = load_image_from_s3(img).convert("RGB")  # type: ignore
            # only keep the second half of the image... non mask part
            image = image.crop((image.width // 2, 0, image.width, image.height))

            input_image = image
            if image.size != config.default_in_shape:
                input_image = image.resize(
                    (config.default_in_shape[0], config.default_in_shape[1]),
                    Image.BICUBIC,
                )

            input_tensor = format_input(input_image)
            fused_mask_tensor = self.model(input_tensor, Image.BICUBIC)[0][0]  # type: ignore
            output_mask = np.asarray(fused_mask_tensor)

            if image.size != config.default_in_shape:
                output_mask = cv2.resize(output_mask, dsize=image.size)

            output_mask = np.tile(np.expand_dims(output_mask, axis=2), [1, 1, 3])
            output_image = np.expand_dims(np.array(image) / 255.0, 0)[0]
            if config.apply_eval_mask:
                output_image = apply_mask(output_image, output_mask)
            else:
                output_image = output_mask

            if config.merge_eval:
                output_image = np.concatenate((output_mask, output_image), axis=1)

            output_image = cv2.cvtColor(output_image.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0  # type: ignore
            output_location = config.output_dir.joinpath(pathlib.Path(img).name)
            cv2.imwrite(str(output_location), output_image)
