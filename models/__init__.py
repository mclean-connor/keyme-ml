# import segmentation models
from models.segmentation.U2Net.model import U2Net
from models.segmentation.BittingHighRes.model import BittingHighRes
from models.segmentation.KeyPresence.model import KeyPresence

# import classification models
from models.classification.BittingFlatMetalDetectorModel.model import (
    BittingFlatMetalDetector,
)


# define the available models
models = {
    "U2Net": U2Net,
    "BittingHighRes": BittingHighRes,
    "KeyPresence": KeyPresence,
    "BittingFlatMetalDetector": BittingFlatMetalDetector,
}
