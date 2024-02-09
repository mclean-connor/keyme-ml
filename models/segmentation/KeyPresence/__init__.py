import pathlib
from models.segmentation.U2Net import U2NetConfig


class LeftConfig(U2NetConfig):
    # Dataset
    dataset_dir: pathlib.Path = pathlib.Path("training-data/bitting_left")


class RightConfig(U2NetConfig):
    # Dataset
    dataset_dir: pathlib.Path = pathlib.Path("training-data/bitting_right")
