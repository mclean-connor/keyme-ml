import pathlib
from models.base_model import BaseModelConfig


class ICNetConfig(BaseModelConfig):
    # Training
    resume: bool = False
    batch_size: int = 24
    epochs: int = 20
    learning_rate: float = 0.0001

    # Dataset
    dataset_dir: pathlib.Path = pathlib.Path("training-data/bitting_left")

    # Model
    default_in_shape: tuple = (512, 1024, 3)
    default_out_shape: tuple = (default_in_shape[0], default_in_shape[1], 1)

    # evaluate
    apply_eval_mask: bool = True
    merge_eval: bool = True
