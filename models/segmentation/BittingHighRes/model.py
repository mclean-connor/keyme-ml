from models.segmentation.U2Net.model import U2Net
from models.segmentation.BittingHighRes import LeftConfig, RightConfig


class BittingHighResLeft(U2Net):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="BittingHighResLeft",
            tiny_model=False,
            config=LeftConfig(),
            *args,
            **kwargs
        )


class BittingHighResRight(U2Net):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="BittingHighResRight",
            tiny_model=False,
            config=RightConfig(),
            *args,
            **kwargs
        )
