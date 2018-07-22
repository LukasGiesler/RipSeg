import cv2
from enum import Enum
import math

# All connection lines between tfpose joints
CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]

# All tfpose joints
class BodyPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

# Weights used for distance_to_point method
BodyPartWeights = [1, 1, 1, 1.2, 1.2, 1, 1.2, 1.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # weights lower than 1 will extend the radius around the joint

# Colors used for segementation, joints and connection lines
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

class VectorMath:
    """
    Helper class for math functions
    """
    @staticmethod
    def distance_point(p0,p1):
        diff = (p1[0] - p0[0], p1[1] - p0[1])
        length = math.sqrt(diff[0]**2 + diff[1]**2)
        return length

    @staticmethod
    def lineMagnitude (x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
        return lineMagnitude

    @staticmethod
    def distance_to_line(p0, p1, p2):
        px = p0[0]
        py = p0[1]
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        LineMag = VectorMath.lineMagnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            ix = VectorMath.lineMagnitude(px, py, x1, y1)
            iy = VectorMath.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = VectorMath.lineMagnitude(px, py, ix, iy)

        return DistancePointLine

class BodySegmentation:
    """
    Segmentation Data for a single body/human in the image
    """

    def __init__(self, width, height, human):
        super().__init__()
        self.mask = []
        self.keypoints = {}
        self.human = human
        self.width = width
        self.height = height

    def calculate_pixel_keypoints(self, human):
        """
        :param human: human to calculate the unnormalized keypoints for
        :return: keypoints object with index corresponding to the bodypart
        """
        self.keypoints = {}
        for i in range(BodyPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            x = int(self.width*body_part.x + 0.5)
            y = int(self.height*body_part.y + 0.5)
            self.keypoints[i] = (x, y)

        return self.keypoints

    def apply_to_mask(self, masks, idx):
        """
        apply this body to the given mask
        :param mask: the mask
        :param idx: idx for the mask that applies to this body
        :return: new segmentation mask for the image
        """
        masks = masks.astype(int)

        for row in range(len(masks)):
            for col in range(len(masks[row])):
                if masks[row][col][idx]:
                    # col and row is swapped for mask to image (y,x) -> (x,y)
                    masks[row][col][idx] = self.calculate_joint_for_pixel((col, row))

        self.mask = masks[:, :, idx]
        return masks

    def calculate_joint_for_pixel(self, p0, fusion=True):
        """
        return the joint the pixel should be applied to
        :param p0: the pixel/coordinate to calculate the joint for
        :return: int bodypart id to apply the pixel to
        """

        nearest = 99999
        body_part = -1

        for i in range(BodyPart.Background.value):
            if i not in self.human.body_parts.keys() or CocoPairs[i][0] not in self.keypoints.keys() or CocoPairs[i][1] not in self.keypoints.keys():
                continue

            #distance = VectorMath.distance_point(p0, self.keypoints[i])
            p1 = self.keypoints[CocoPairs[i][0]]
            p2 = self.keypoints[CocoPairs[i][1]]
            distance = VectorMath.distance_to_line(p0, p1, p2)
            #distance = distance * BodyPartWeights[i];

            if distance < nearest:
                #print("p0: " + str(p0) + " A: " + str(p1) + " B: " + str(p2) + " Distance: " + str(distance))
                nearest = distance
                # we need numbers from 1 upwards
                body_part = i+1
                # fuse body parts of head and upper body, all body parts are index + 1 here!
                if(fusion):
                    if(body_part == 13 or body_part == 14 or body_part == 15 or body_part == 16 or body_part == 17 or body_part == 18 or body_part == 19):
                        body_part = 13
                    if(body_part == 7 or body_part == 10):
                        body_part = 10

        return body_part


class RipSeg:
    """
    processes the output from openpose and mrcnn to create a segmentation mask
    """

    def __init__(self,width, height):
        super().__init__()
        self.bodies = []
        self.width = width
        self.height = height

    # calculate the body segments from the tfpose and mrcnn output
    def segment_bodies(self, humans, mrcnn):
        r = mrcnn[0]
        # rois = r['rois']
        masks = r['masks']
        # class_ids = r['class_ids']
        # scores = r['scores']

        for i in range(len(humans)):
            self.bodies.append(self.segment_body(humans[i]))
            # TODO: index of tfpose human array and mrcnn masks are probably not the same
            masks = self.bodies[-1:][0].apply_to_mask(masks, i)

        return masks

    def segment_body(self, human):
        """
        create the pixel keypoints to segment a single human
        :param human: the human to segment
        :return: the updated segmentation mask
        """
        body = BodySegmentation(self.width, self.height, human)
        body.calculate_pixel_keypoints(human)
        return body
