if __name__ == '__main__':
    """
        test for simple image
    """
    import os
    import sys
    import random
    import math
    import numpy as np
    import skimage.io
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library

    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize_cv

    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "../samples/coco/"))  # To find local version
    import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "../images")

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = [
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard'
    ]

    filename = os.path.join(IMAGE_DIR,'snowboard.jpg')
    image = skimage.io.imread(filename)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    frame = visualize_cv.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )
    cv2.imshow("frame",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    