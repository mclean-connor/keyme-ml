from models.segmentation.U2Net.model import U2Net

class KeyPresence(U2Net):
    def __init__(self):
        super().__init__(name='KeyPresence', tiny_model=True)
