from typing import List
from tensorflow import keras
import gc
import wandb

from models.utils import get_image_paths
from sklearn.model_selection import train_test_split
from models import models
from models.base_model import BaseKeyMeModel


def train_and_evaluate_model(
    model: BaseKeyMeModel, train_images: List[str], eval_images: List[str]
):

    model.train(train_images, eval_images)

    return "Training and evaluation complete"


def train_new_model(model_name: str):
    # make sure the model name is valid
    if model_name not in models.keys():
        raise ValueError(f"Model {model_name} not found")

    # get the model to train
    trainType = type(models[model_name])
    if trainType == list:
        models_to_train = models[model_name]
    else:
        models_to_train = [models[model_name]]

    # train the models
    for m in models_to_train:
        # initialize the model
        model = m()

        # get the training and evaluation images
        data_path = str(model.config.dataset_dir)
        all_images = get_image_paths(data_path)
        train_images, eval_images = train_test_split(
            all_images, test_size=0.15, random_state=42
        )

        # notifify where run can be tracket
        run_url = wandb.run.get_url()  # type: ignore
        print(f"Track training and evaluation at: {run_url}")

        # run traning script
        print(f"Training and evaluating new {model.name} model")
        output = train_and_evaluate_model(model, train_images, eval_images)

        # these are needed to dealocate memory used by the model
        keras.backend.clear_session()
        del model
        gc.collect()
