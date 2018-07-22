import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

class TFPoseController:
    """
    tfpose controler
    """
    def __init__(self,model,resolution,scales):
        self.scales = scales
        self.width, self.height = model_wh(resolution)
        self.estimator = TfPoseEstimator(get_graph_path(model), target_size=(self.width, self.height))

    # return the calculated output from OpenPose for the image
    def applyImage(self,image):
        humans = self.estimator.inference(image, self.scales)
        return humans

    # return image of paf in x direction
    def draw_x_paf(self):
        vectormap = self.estimator.pafMat.transpose((2, 0, 1))
        vectormapx = np.amax(np.absolute(vectormap[::2, :, :]), axis=0)
        return vectormapx

    # return image of paf in y direction
    def draw_y_paf(self):
        vectormap = self.estimator.pafMat.transpose((2, 0, 1))
        vectormapy = np.amax(np.absolute(vectormap[1::2, :, :]), axis=0)
        return vectormapy

    # return image of heatmap
    def draw_heatmap(self):
        return np.amax(self.estimator.heatMat[:, :, :-1], axis=2)

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False):
        return TfPoseEstimator.draw_humans(npimg,humans,imgcopy)
