from prefect import flow, task, get_run_logger
import subprocess
from typing import List
from tensorflow import keras
import gc
import wandb

from models.utils import get_image_paths
from sklearn.model_selection import train_test_split
from models import models
from models.base_model import BaseKeyMeModel
from update_dvc import sync_dvc_data
from utils import run_subprocess


def create_message(message: str, status: str):
    return {"status": status, "message": message}


@task
def train_and_evaluate_model(
    model: BaseKeyMeModel, train_images: List[str], eval_images: List[str]
):
    model.train(train_images, eval_images)

    return "Training and evaluation complete"


@task
def commit_and_push_git():
    # Check for changes before committing
    status_result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if status_result.stdout.strip():  # If there's output, there are changes
        run_subprocess(["git", "add", "."], "Adding changes to git")
        run_subprocess(
            ["git", "commit", "-m", "Update training data version."],
            "Committing changes to git",
        )
        run_subprocess(["git", "push"], "Pushing changes to git")
    else:
        print("No changes to commit to git.")


@flow
def train_new_model(model_name: str):
    logger = get_run_logger()

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
        all_images = get_image_paths(model.config.dataset_dir)
        train_images, eval_images = train_test_split(
            all_images, test_size=0.15, random_state=42
        )

        # notifify where run can be tracket
        run_url = wandb.run.get_url()  # type: ignore
        logger.info(f"Track training and evaluation at: {run_url}")

        # run traning script
        logger.info(f"Training and evaluating new {model.name} model")
        output = train_and_evaluate_model(model, train_images, eval_images)

        # these are needed to dealocate memory used by the model
        keras.backend.clear_session()
        del model
        gc.collect()

    # commit_and_push_git()


if __name__ == "__main__":
    train_new_model("BittingHighResRight")
