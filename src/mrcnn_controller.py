
import sys
import os
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import MaskRCNN
sys.path.append(os.path.join(ROOT_DIR, "./mrcnn/"))  # To find local version
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "../samples/coco/"))  # To find local version
 # Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "../logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)    

import mrcnn.model as modellib
from mrcnn import utils
import coco

class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

class MRcnnController(coco.CocoConfig):
    """
    mrcnn controller
    """

    def __init__(self):
        config = InferenceConfig()
        config.display()

        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Run detection
    def getMaskData(self,image):
        results = self.model.detect([image], verbose=1)
        return results    