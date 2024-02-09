from models.segmentation.U2Net.model import U2Net
from models.segmentation.KeyPresence import LeftConfig, RightConfig


class KeyPresenceLeft(U2Net):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            name="KeyPresenceLeft",
            tiny_model=True,
            config=LeftConfig(),
            *args,
            **kwargs,
        )


class KeyPresenceRight(U2Net):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            name="KeyPresenceRight",
            tiny_model=True,
            config=RightConfig(),
            *args,
            **kwargs,
        )
