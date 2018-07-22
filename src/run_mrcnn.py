import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mrcnn_controller import MRcnnController


if __name__ == '__main__':

    # Root directory of the project
    ROOT_DIR = os.path.abspath("./")

    # Import TFPose
    sys.path.append(ROOT_DIR)  # To find local version of the library

    # Import TFPose
    sys.path.append(os.path.join(ROOT_DIR, "./tfpose/"))  # To find local version
    # Import MaskRCNN
    sys.path.append(os.path.join(ROOT_DIR, "./mrcnn/"))  # To find local version

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "../images")

    from mrcnn import visualize_cv

    class_names = [
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard'
    ]

    controller = MRcnnController()
    filename = os.path.join(IMAGE_DIR,'natural_pose.jpg')
    print("filename: " + str(filename))
    image = skimage.io.imread(filename)
    results = controller.getMaskData(image)

    # Visualize results
    r = results[0]
    frame = visualize_cv.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )
    cv2.imshow("frame",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()