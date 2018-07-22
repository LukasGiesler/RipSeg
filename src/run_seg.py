import os
import sys
import random
import math
import argparse
import logging
import time
import ast
import numpy as np
import skimage.io
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mrcnn_controller import MRcnnController
from tkinter import filedialog as fd
import PIL.Image, PIL.ImageTk
import tkinter
from body_segmentation import RipSeg
from cv_seg_visualizer import RipSegVisualizer

logger = logging.getLogger('DEBUG')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def blur_image():
    print("blurred")

def ask_for_file():
    global photo
    global filename
    filename = fd.askopenfilename()
    cv2_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    height, width, no_channels = cv2_img.shape
    canvas = tkinter.Canvas(window, width=width, height=height)
    canvas.grid(row=0, column=2, columnspan=20, rowspan=20,
               sticky=tkinter.W+tkinter.E+tkinter.N+tkinter.S)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2_img))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

def run_segmentation():
    global photo
    global photo_cropped
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default=filename)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--lines', type=int, default=label_lines_entry_text.get(), help='draw lines between joints. default=0')
    parser.add_argument('--keypoints', type=int, default=label_keypoints_entry_text.get(), help='draw joints. default=0')
    parser.add_argument('--crop', type=int, default=label_crop_entry_text.get(), help='draw cropped image. default=0')
    parser.add_argument('--alpha', type=float, default=label_alpha_entry_text.get(), help='sets the alpha of the RipSeg mask. default=1.0')

    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    totalTime = time.time()

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

    # import needed modules from tfpose and MaskRCNN
    import common
    from tfpose_controller import TFPoseController

    from lifting.prob_model import Prob3dPose
    from lifting.draw import plot_pose
    from mrcnn import visualize_cv

    # finally apply tfpose and maskrcnn
    tfposeController = TFPoseController(args.model, args.resolution, scales)

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    t = time.time()

    tfPoseOutput = tfposeController.applyImage(image)

    elapsed = time.time() - t

    logger.info('Step (1/3): OpenPose completed: %s in %.4f seconds.' % (args.image, elapsed))
    message_openpose='OpenPose completed in: %.4f seconds.' % (elapsed)
    label_text_openpose_runtime.set(message_openpose)
    t = time.time()

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

    controller = MRcnnController()
    image = skimage.io.imread(args.image)

    mrcnnOutput = controller.getMaskData(image)

    elapsed = time.time() - t
    logger.info('Step (2/3) MASK R-CNN completed: %s in %.4f seconds.' % (args.image, elapsed))
    message_mask = 'MASK R-CNN completed in: %.4f seconds.' % (elapsed)
    label_text_maskrcnn_runtime.set(message_mask)
    t = time.time()

    image_h, image_w, channels = image.shape
    ripSeg = RipSeg(image_w, image_h)
    ripSeg.segment_bodies(tfPoseOutput, mrcnnOutput)
    image = RipSegVisualizer.draw_ripseg(image, ripSeg, args.lines, args.keypoints, args.crop, args.alpha)
    canvas = tkinter.Canvas(window, width=image_w, height=image_h)
    canvas.grid(row=0, column=2, columnspan=15, rowspan=15,
               sticky=tkinter.W+tkinter.E+tkinter.N+tkinter.S)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

    if args.crop == 1:
        cropped = cv2.imread('cropped.png')
        cropped_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_h, cropped_w, cropped_channels = cropped_img.shape
        canvas = tkinter.Canvas(window, width=cropped_w, height=cropped_h)
        canvas.grid(row=0, column=17, columnspan=4, rowspan=4,
                   sticky=tkinter.W+tkinter.E+tkinter.N+tkinter.S)
        photo_cropped = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cropped_img))
        canvas.create_image(0, 0, image=photo_cropped, anchor=tkinter.NW)

    elapsed = time.time() - t
    logger.info('Step (3/3) Body Segmentation completed: %s in %.4f seconds.' % (args.image, elapsed))
    message_body='Body Segmentation completed in: %.4f seconds.' % (elapsed)
    label_text_body_segmentation_runtime.set(message_body)
    elapsed = time.time() - totalTime
    logger.info('RipSeg completed in %.4f seconds.' % (elapsed))
    message_ripseg='RipSeg completed in: %.4f seconds.' % (elapsed)
    label_text_ripseg_runtime.set(message_ripseg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    window = tkinter.Tk()
    window.title("RipSeg")
    message_openpose = "OpenPose completed in:"
    message_mask="MASK R-CNN completed in:"
    message_body = "Body Segmentation completed in:"
    message_ripseg = "RipSeg completed in:"
    btn_select_photo = tkinter.Button(window, text="Select Image", width=10, command=ask_for_file)
    btn_select_photo.grid(row=0, column=0)
    btn_run_seg = tkinter.Button(window, text ="Run Segmentation",width=15, command=run_segmentation)
    btn_run_seg.grid(row=0, column=1)
    label_text_openpose_runtime = tkinter.StringVar()
    label_text_maskrcnn_runtime = tkinter.StringVar()
    label_text_openpose_runtime.set(message_openpose)
    label_text_maskrcnn_runtime.set(message_mask)
    label_text_body_segmentation_runtime = tkinter.StringVar()
    label_text_body_segmentation_runtime.set(message_body)
    label_text_ripseg_runtime = tkinter.StringVar()
    label_text_ripseg_runtime.set(message_ripseg)
    label_mask = tkinter.Label(window, textvariable=label_text_maskrcnn_runtime)
    label_openpose = tkinter.Label(window, textvariable=label_text_openpose_runtime)
    label_body = tkinter.Label(window, textvariable=label_text_body_segmentation_runtime)
    label_ripseg = tkinter.Label(window, textvariable=label_text_ripseg_runtime)
    label_openpose.grid( row=5,column=0)
    label_mask.grid( row=6,column=0)
    label_body.grid(row=7,column=0)
    label_ripseg.grid(row=8,column=0)

    tkinter.Label(window, text="Lines 0 or 1").grid(row=1, column=0)
    entry_lines_default = 0
    label_lines_entry_text = tkinter.IntVar()
    label_lines_entry_text.set(entry_lines_default)
    entry_lines = tkinter.Entry(window, textvariable=label_lines_entry_text)
    entry_lines.grid(row=1, column=1)

    tkinter.Label(window, text="Keypoints 0 or 1").grid(row=2, column=0)
    entry_keypoints_default = 0
    label_keypoints_entry_text = tkinter.IntVar()
    label_keypoints_entry_text.set(entry_keypoints_default)
    entry_keypoints = tkinter.Entry(window, textvariable=label_keypoints_entry_text)
    entry_keypoints.grid(row=2, column=1)

    tkinter.Label(window, text="Crop 0 or 1").grid(row=3, column=0)
    entry_crop_default = 0
    label_crop_entry_text = tkinter.IntVar()
    label_crop_entry_text.set(entry_crop_default)
    entry_crop = tkinter.Entry(window, textvariable=label_crop_entry_text)
    entry_crop.grid(row=3, column=1)

    tkinter.Label(window, text="Alpha between 0 - 1.0 (Double)").grid(row=4, column=0)
    entry_alpha_default = 0.5
    label_alpha_entry_text = tkinter.DoubleVar()
    label_alpha_entry_text.set(entry_alpha_default)
    entry_alpha = tkinter.Entry(window, textvariable=label_alpha_entry_text)
    entry_alpha.grid(row=4, column=1)


    cv_img_legende = cv2.cvtColor(cv2.imread("RipSeg-Legende.png"), cv2.COLOR_BGR2RGB)
    height, width, no_channels = cv_img_legende.shape
    canvas_legende = tkinter.Canvas(window, width=width, height=height)
    canvas_legende.grid(row=9, column=0, columnspan=2, rowspan=15)
    photo_legende = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img_legende))
    canvas_legende.create_image(0, 0, image=photo_legende, anchor=tkinter.NW)

    window.mainloop()
