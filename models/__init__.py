# import segmentation models
from models.segmentation.U2Net.model import U2Net
from models.segmentation.BittingHighRes.model import (
    BittingHighResLeft,
    BittingHighResRight,
)
from models.segmentation.KeyPresence.model import KeyPresenceLeft, KeyPresenceRight

# import classification models
from models.classification.BittingFlatMetalDetectorModel.model import (
    BittingFlatMetalDetector,
)


# define the available models
models = {
    "U2Net": U2Net,
    "BittingHighResLeft": BittingHighResLeft,
    "BittingHighResRight": BittingHighResRight,
    "KeyPresenceLeft": KeyPresenceLeft,
    "KeyPresenceRight": KeyPresenceRight,
    "BittingFlatMetalDetector": BittingFlatMetalDetector,
    "BittingModels": [
        BittingHighResLeft,
        BittingHighResRight,
        KeyPresenceLeft,
        KeyPresenceRight,
    ],
}
