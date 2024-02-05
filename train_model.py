from prefect import flow, task
import subprocess

from models.utils import list_s3_dvc_files, tf2_to_onnx
from sklearn.model_selection import train_test_split
from models import models


def run_subprocess(command, task_name=""):
    print(f"Running {task_name}...")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error in {task_name}: {result.stderr}")
    if result.returncode != 0:
        raise Exception(f"{task_name} failed")


@task
def update_dvc():
    run_subprocess(
        ["dvc", "update", "--no-download", "training-data.dvc"], "Pulling DVC data"
    )


@task
def commit_and_push_new_data():
    run_subprocess(["dvc", "commit"], "Committing DVC")
    run_subprocess(["dvc", "push"], "Pushing DVC")


@flow
def sync_dvc_data():
    update_dvc()
    commit_and_push_new_data()


@task
def run_training(model, images):
    print("Training the model")
    model.train(images)

    if model.use_onnx:
        print("Converting the model to ONNX")
        onnx_path = f"{model.config.run_dir}/onnx_weights.onnx"
        tf2_to_onnx(model.model, onnx_path)

    return model


@task
def run_evaluation(model, images):
    print("Evaluating the model")
    model.evaluate(images)

    return model


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
def train_model(training_args: dict):
    model_name = training_args["model_name"]
    model = models[model_name]()

    sync_dvc_data()

    all_images = list_s3_dvc_files(model.config.bucket, model.config.prefix)
    train_images, eval_images = train_test_split(
        all_images, test_size=0.025, random_state=42
    )

    trained_model = run_training(model, train_images)

    run_evaluation(trained_model, eval_images)

    commit_and_push_git()


if __name__ == "__main__":
    train_model({"model_name": "BittingHighRes"})
