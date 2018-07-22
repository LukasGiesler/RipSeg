import argparse
import logging
import time
import ast

import common
import cv2
import numpy as np
import matplotlib.pyplot as plt
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

import visualize_cv

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='../images/p1.jpg')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

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

    for human in humans:
        print(human)
    
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

    #maskImg = visualize_cv.display_masks(bgimg,vectormapx,vectormapy)
    #maskImg = cv2.resize(maskImg, (srcImage.shape[1], srcImage.shape[0]))

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