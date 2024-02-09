from dotenv import load_dotenv

load_dotenv()  # This loads the environment variables from .env

import os
import wandb
from typing import List, Union
import pathlib
import datetime
from pydantic import BaseModel
from tensorflow import keras
import cv2
import numpy as np
import tf2onnx
import wandb

base_dir = pathlib.Path(__file__).parent.parent
base_weight_dir = base_dir.joinpath("weights")


# define the configuration for the model
class BaseModelConfig(BaseModel):
    # save weights based on timestamp
    name: str = "weights"
    run_id: str = ""

    # Dataset
    dataset_dir: pathlib.Path = base_dir.joinpath("training_data")
    eval_dir: pathlib.Path = (
        base_dir.joinpath("outputs").joinpath(name).joinpath(run_id)
    )

    # Model
    default_in_shape: tuple = (512, 512, 3)
    default_out_shape: tuple = (default_in_shape[0], default_in_shape[1], 1)

    # Training
    resume: bool = False
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 0.001


# Set up Weights and Biases
wandb.login(key=os.environ["WANDB_API_KEY"])


class BaseKeyMeModel:
    def __init__(self, name: str, config: BaseModelConfig, training: bool = True):
        self.name = name
        self.config = config
        self.config.name = name
        self.model: keras.Model
        self.use_onnx = True
        self.run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.weights_dir = base_weight_dir
        # Set up Weights and Biases
        if training:
            wandb.init(project=name, name=self.run_id)
            # when training, save the weights to wandb
            self.weights_dir = os.path.join(wandb.run.dir, "weights")  # type: ignore

    def train(self, train_imgs: List[str], eval_imgs: List[str]) -> None:
        """
        this is the main training loop. It should be called from the main thread.
        to edit the training loop, override the _train method
        """
        # train the model
        self._train(train_imgs, eval_imgs)

        # covnert the model to onnx if specified
        if self.use_onnx:
            self._convert_to_onnx()

        wandb.finish()
        while wandb.run is not None:
            pass

    def _train(self, train_imgs: List[str], eval_imgs: List[str]) -> None:
        """
        Train the model on the given images
        """
        raise NotImplementedError(
            "Train method not implemented. Please override the _train method."
        )

    def predict(self, image_path: str, weights_path: Union[str, None]) -> np.ndarray:
        """
        this is the main prediction loop. It should be called from the main thread.
        to edit the prediction loop, override the _predict method
        """
        if weights_path:
            self._load_weights(weights_path)

        return self._predict(image_path)

    def _predict(self, image_path: str) -> np.ndarray:
        """
        Predict the mask for the given image
        """
        raise NotImplementedError(
            "Predict method not implemented. Please override the _predict method."
        )

    def _save_eval_image(
        self, img: np.ndarray, file_name: str, to_wandb: bool = True
    ) -> None:
        """
        Save the evaluation image
        """
        # make sure the evaluation directory exists
        os.makedirs(self.config.eval_dir, exist_ok=True)

        # save the image
        img_path = self.config.eval_dir.joinpath(file_name)
        cv2.imwrite(str(img_path), img)

        # backup the image to wandb
        if to_wandb:
            wandb.log({"eval_pred": wandb.Image(img)})

    def _save_weights(self, top_n: int = 2) -> None:
        """
        Save the model weights
        """
        # save the weights
        # get current time stamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_save_path = os.path.join(self.weights_dir, f"{self.name}-{timestamp}.h5")

        # make sure the weights directory exists
        os.makedirs(self.weights_dir, exist_ok=True)

        # list all saved h5 models and keep only the top_n versions
        saved_weights = list(pathlib.Path(self.weights_dir).glob("*.h5"))
        if len(saved_weights) >= top_n:
            oldest = min(saved_weights, key=os.path.getctime)
            os.remove(oldest)

        # save the model to wandb... it will be uploaded at end of run
        self.model.save_weights(model_save_path)

    def _convert_to_onnx(self) -> None:
        """
        Convert the model to ONNX format. This is a lenghty process and should be run
        after the training completes.
        """
        onnx_dir = os.path.join(self.weights_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)

        # check the h5 files in the weights directory and convert them to onnx
        for h5_file_path in pathlib.Path(self.weights_dir).glob("*.h5"):
            print(f"Converting {h5_file_path} to ONNX")
            onnx_file_path = os.path.join(onnx_dir, f"{h5_file_path.stem}.onnx")

            # load the model weights
            self.model.load_weights(h5_file_path)

            # convert the model to onnx
            _, _ = tf2onnx.convert.from_keras(self.model, output_path=onnx_file_path)

    def _load_weights(self, weights_path: str) -> None:
        """
        Load the weights from the given path
        """
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
