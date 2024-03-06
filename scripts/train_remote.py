import modal
from modal import Image, Secret
import os
import dotenv

# import the training script
from models.training import train_new_model

# load the .env file
dotenv.load_dotenv()

# init variables needed for Modal
stub = modal.Stub("vision-model-training")
image = (
    Image.from_registry(
        "tensorflow/tensorflow:2.14.0-gpu",
    )
    .poetry_install_from_file("pyproject.toml")
    .apt_install("ffmpeg", "libsm6", "libxext6")
)
secrets = [Secret.from_dotenv(".env")]


@stub.function(
    timeout=int(os.environ.get("TRAINING_TIMEOUT", 60 * 60 * 6)),  # default 6 hours
    gpu="A10G",
    image=image,
    secrets=secrets,
    volumes={
        "/training-data": modal.S3Mount(
            os.environ.get("S3_TRAINING_DATA_BUCKET", ""),
            secret=Secret.from_dotenv(".env"),
            read_only=True,
        )  # type: ignore
    },
)
def start_training(model_name: str):
    # run the training script
    train_new_model(model_name)


@stub.local_entrypoint()
def main(model: str):
    train_new_model.remote(model)
