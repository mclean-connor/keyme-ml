import pathlib
from models.base_model import BaseModelConfig


class MillingConfig(BaseModelConfig):
    # Training
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "sparse_categorical_crossentropy"
    metrics: list = ["accuracy"]

    # Dataset
    dataset_dir: pathlib.Path = pathlib.Path("training-data/bitting_left")

    # Model
    default_in_shape: tuple = (224, 224, 3)
    use_third_convolutional_block: bool = False
    y_to_label: dict = {}
