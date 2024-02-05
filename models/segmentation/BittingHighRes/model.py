from models.segmentation.U2Net.model import U2Net


class BittingHighRes(U2Net):
    def __init__(self):
        super().__init__(name="BittingHighRes", tiny_model=False)
