from dotenv import load_dotenv

load_dotenv()  # This loads the environment variables from .env

import os
import wandb
from typing import List
import pathlib
import datetime
from pydantic import BaseModel
from tensorflow import keras
from sklearn.model_selection import train_test_split
from prefect import task, flow

# project specific imports
from models.utils import tf2_to_onnx, list_s3_objects

base_dir = pathlib.Path(__file__).parent.parent
weights_dir = base_dir.joinpath("weights")

# if the weights directory does not exist, create it
if not weights_dir.exists():
    weights_dir.mkdir()

run_id: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
run_dir_path: pathlib.Path = weights_dir.joinpath(run_id)

# if the run directory does not exist, create it
if not run_dir_path.exists():
    run_dir_path.mkdir()


# define the configuration for the model
class BaseModelConfig(BaseModel):
    # save weights based on timestamp
    name: str = "weights"
    weight_dir: pathlib.Path = weights_dir
    run_dir: pathlib.Path = run_dir_path
    weights_file: pathlib.Path = run_dir.joinpath(name).joinpath(f"{name}.h5")

    # Dataset
    output_dir: pathlib.Path = (
        base_dir.joinpath("outputs").joinpath(name).joinpath(run_id)
    )

    # s3 bucket and prefix for training data
    bucket: str = "keyme-data"
    prefix: str = "bitting_left"

    # Training
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 0.001


# Set up Weights and Biases
wandb.login(key=os.environ["WANDB_API_KEY"])


class BaseKeyMeModel:
    def __init__(self, name: str, config: BaseModelConfig):
        self.name = name
        self.config = config
        self.config.name = name
        self.model: keras.Model
        self.use_onnx = True
        # Set up Weights and Biases
        wandb.init(project=name)

    def train(self, imgs: List[str]) -> None:
        """
        Train the model on the given images
        """
        self.model.fit(
            imgs, epochs=self.config.epochs, batch_size=self.config.batch_size
        )

    def evaluate(self, imgs: List[str]) -> None:
        """
        Evaluate the model on the given images
        """
        self.model.evaluate(imgs)
        pass
