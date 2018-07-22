import os
import sys
import argparse
import logging
import time
import ast
import cv2
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
    parser.add_argument('--image', type=str, default='../images/natural_pose.jpg')
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

    #import needed modules from tfpose and MaskRCNN
    import common
    from tfpose_controller import TFPoseController

    from lifting.prob_model import Prob3dPose
    from lifting.draw import plot_pose

    #finally apply tfpose and maskrcnn
    tfposeController = TFPoseController(args.model,args.resolution,scales)

    # estimate human poses from a single image !
    srcImage = common.read_imgfile(args.image, None, None)
    image = common.read_imgfile(args.image, None, None)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    t = time.time()

    humans = tfposeController.applyImage(image)

    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TFPoseController.draw_humans(image, humans, imgcopy=False)

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')

    # result image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #resize image for heatmap and vectormap
    bgimg = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (tfposeController.estimator.pafMat.shape[1], tfposeController.estimator.pafMat.shape[0]))

    #maskImg = visualize_cv.display_masks(bgimg,vectormapx,vectormapy)
    #maskImg = cv2.resize(maskImg, (srcImage.shape[1], srcImage.shape[0]))

    # heatmap
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg)
    plt.imshow(tfposeController.draw_heatmap(),alpha=0.5)
    plt.colorbar()

    # vectormap x
    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    plt.imshow(bgimg)
    plt.imshow(tfposeController.draw_x_paf(),alpha=0.5)
    plt.colorbar()

    # vectormap y
    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    plt.imshow(bgimg)
    plt.imshow(tfposeController.draw_y_paf(),alpha=0.5)
    plt.colorbar()

    plt.show()
    pass
