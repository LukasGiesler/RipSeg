import cv2
from body_segmentation import CocoPairsRender

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [100, 255, 100],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

class RipSegVisualizer:

    @staticmethod
    def draw_ripseg(image, ripseg, drawLines=0.0, drawKeypoints=0.0, crop=0.0, alpha=1.0):
        """
        Draws the ripseg output into the image
        :param image: the input image
        :param ripseg:
        :return:
        """
        for body in ripseg.bodies:
            if(crop==1):
                RipSegVisualizer.draw_croppedBodyPart(image, body.mask)

            image = RipSegVisualizer.draw_mask(image, body.mask, alpha)

            if(drawLines==1):
                RipSegVisualizer.draw_joint_lines(image, body.human, body.keypoints)
            if(drawKeypoints==1):
                RipSegVisualizer.draw_joint_keypoints(image, body.keypoints)

        return image

    @staticmethod
    def draw_joint_keypoints(image, keypoints, radius=3):
        if (radius <= 0.0):
            image
        for idx in keypoints:
            cv2.circle(image, keypoints[idx], radius, CocoColors[idx], -1)
        return image

    @staticmethod
    def draw_joint_lines(image, human, keypoints, linewidth=3):
        """
        draws joint lines into the image
        :param image: input image
        :param human: human to draw joint lines for
        :param keypoints: the keypoints of the human
        :return:
        """
        if (linewidth <= 0.0):
            return image
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            cv2.line(image, keypoints[pair[0]], keypoints[pair[1]], CocoColors[pair_order], linewidth)
        return image

    def draw_mask(image, mask, alpha=1):
        """
        draw the mask for the segmented body into the image
        :param image: input image
        :param mask: single mask for the body as 2d array
        :param alpha: [optional] alpha value
        :return:
        """
        for row in range(len(mask)):
            for col in range(len(mask[row])):
                if mask[row][col] > 0:
                    for n in range(3):
                        image[row][col][n] = image[row][col][n] * (1-alpha) + alpha * CocoColors[mask[row][col]-1][n]

        return image

    def draw_croppedBodyPart(image, mask):
        """
        draw a single body part as seperate image
        :param image: input image
        :param mask: single mask for the body as 2d array
        :return:
        """
        #TODO: Implement cropping of selected body part e.g. head
        LowestX = 99999999
        HighestX = 0
        LowestY = 99999999
        HighestY = 0
        for row in range(len(mask)):
            for col in range(len(mask[row])):
                if mask[row][col] > 0:
                    for n in range(3):
                        if((mask[row][col]-1) == 13):# Draw Rectangle around head for cropping purposes
                            if(row < LowestY):
                                LowestY = row
                            if(row > HighestY):
                                HighestY = row
                            if(col < LowestX):
                                LowestX = col
                            if(col > HighestX):
                                HighestX = col

        startX = LowestX
        endX = HighestX
        startY = LowestY
        endY = HighestY

        croppedImage = image[startY:endY, startX:endX]#startY:endY, startX:endX
        croppedImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB)
        cv2.imwrite('cropped.png', croppedImage)

        return image
