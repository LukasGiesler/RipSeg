import os
import sys
import argparse
import logging
import time
import ast
import math
import skimage.io
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='../images/tpose_mocap.jpg')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    # Root directory of the project
    ROOT_DIR = os.path.abspath("./")

    # Import TFPose
    sys.path.append(ROOT_DIR)  # To find local version of the library

    # Import TFPose
    sys.path.append(os.path.join(ROOT_DIR, "./tfpose/"))  # To find local version
    # Import MaskRCNN
    sys.path.append(os.path.join(ROOT_DIR, "./mrcnn/"))  # To find local version
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "../samples/coco/"))  # To find local version

    #import needed modules from tfpose and MaskRCNN
    import common
    from estimator import TfPoseEstimator
    from networks import get_graph_path, model_wh

    from lifting.prob_model import Prob3dPose
    from lifting.draw import plot_pose

    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize_cv

    import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "../logs")
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "../images")

    #finally apply tfpose and maskrcnn

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    filename = os.path.join(IMAGE_DIR,'tpose_mocap.jpg')
    image = skimage.io.imread(filename)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    frame = visualize_cv.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )

    print(r['rois'])
    print(r['masks'])
    print(r['class_ids'])
    print(r['scores'])
    cv2.imshow("frame",frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    srcImage = common.read_imgfile(args.image, None, None)
    image = common.read_imgfile(args.image, None, None)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    t = time.time()
    humans = e.inference(image, scales=scales)

    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')

    # paf map
    vectormap = e.pafMat.transpose((2, 0, 1))
    vectormapx = np.amax(np.absolute(vectormap[::2, :, :]), axis=0)
    vectormapy = np.amax(np.absolute(vectormap[1::2, :, :]), axis=0)

    # result image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #resize image for heatmap and vectormap
    bgimg = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.pafMat.shape[1], e.pafMat.shape[0]))

    # heatmap
    a = fig.add_subplot(2, 2, 2)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(bgimg)
    plt.imshow(tmp,alpha=0.5)
    plt.colorbar()

    # vectormap x
    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    plt.imshow(bgimg)
    plt.imshow(vectormapx,alpha=0.5)
    plt.colorbar()

    # vectormap x
    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    plt.imshow(bgimg)
    plt.imshow(vectormapy,alpha=0.5)
    plt.colorbar()

    plt.show()
    pass
