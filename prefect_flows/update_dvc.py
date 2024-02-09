from prefect import flow, task
import datetime
import os

from utils import run_subprocess
from models import models
from models.base_model import BaseKeyMeModel


@task
def pull_dvc():
    run_subprocess(["dvc", "pull"], "Pulling DVC data")


@task
def commit_and_push_new_data(model: BaseKeyMeModel):
    new_data_dir = os.environ.get("NEW_DATA_DIR", "new-data")
    # if the directory does not exist or there are no files in it, return
    if not os.path.exists(new_data_dir) or not os.listdir(new_data_dir):
        print("No new data to add to the training data.")
        return

    # add new versioned directory to {training data directory}/{timestamp}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_version_directory = os.path.join(model.config.dataset_dir, timestamp)
    os.makedirs(data_version_directory)

    # copy data from the new-data directory to the new versioned directory
    os.system(f"cp {new_data_dir}/* {data_version_directory}/")

    # add the new versioned directory to DVC
    run_subprocess(
        ["dnc", "add", f"{data_version_directory}"], "Adding DVC file to git"
    )
    run_subprocess(
        ["git", "commit"],
        "Committing changes to dvc",
    )
    # pussh the dvc changes
    run_subprocess(["dvc", "push"], "Pushing changes to dvc")

    # commit and push the new data to git
    run_subprocess(["git", "add", "."], "Adding changes to git")
    run_subprocess(
        ["git", "commit", "-m", "Update training data version."],
        "Committing changes to git",
    )
    run_subprocess(["git", "push"], "Pushing changes to git")


@flow
def sync_dvc_data(model_name: str):
    model = models[model_name](training=False)
    pull_dvc()
    commit_and_push_new_data(model)
