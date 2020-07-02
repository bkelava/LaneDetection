import matplotlib.pylab as plt
import cv2
import numpy as np
import math
from math import atan2, degrees

class DetectionManager():

    def __init__(self):
        pass

    def GetAngleOfLineBetweenTwoPoints(self, x1, y1, x2, y2):
        deltaY = y2 - y1;
        deltaX = x2 - x1;
        result = degrees(atan2(deltaY, deltaX))
        if (deltaY < 0):
            result = result + 360;
        return result

    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([[x1, y1, x2, y2]])


    def average_slope_intercept(self, image, lines):
        img = np.copy(image)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        left_fit = []
        right_fit = []
        right_line = []
        left_line = []
        extra_line_left = []
        extra_line_right = []
        extra_line_left_fit = []
        extra_line_right_fit = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if (self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2, y2) > 30 and self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2,
                                                                                                                y2) < 60):
                        parameters = np.polyfit((x1, x2), (y1, y2), 1)
                        slope = parameters[0]
                        intercept = parameters[1]
                        if slope < 0:
                            left_fit.append((slope, intercept))
                        else:
                            right_fit.append((slope, intercept))
                    if (self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2, y2) > 20 and self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2,
                                                                                                                y2) < 30):
                        parameters = np.polyfit((x1, x2), (y1, y2), 1)
                        slope = parameters[0]
                        intercept = parameters[1]
                        if slope < 0:
                            extra_line_left_fit.append((slope, intercept))
                        else:
                            extra_line_right_fit.append((slope, intercept))
                    if (self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2, y2) > 280 and self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2,
                                                                                                                y2) < 340):
                        parameters = np.polyfit((x1, x2), (y1, y2), 1)
                        slope = parameters[0]
                        intercept = parameters[1]
                        if slope < 0:
                            left_fit.append((slope, intercept))
                        else:
                            right_fit.append((slope, intercept))
                    if (self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2, y2) > 340 and self.GetAngleOfLineBetweenTwoPoints(x1, y1, x2,
                                                                                                                y2) < 350):
                        parameters = np.polyfit((x1, x2), (y1, y2), 1)
                        slope = parameters[0]
                        intercept = parameters[1]
                        if slope < 0:
                            extra_line_left_fit.append((slope, intercept))
                        else:
                            extra_line_right_fit.append((slope, intercept))
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = self.make_coordinates(image, left_fit_average)
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.make_coordinates(image, right_fit_average)
        if extra_line_left_fit:
            extra_left_fit_average = np.average(extra_line_left_fit, axis=0)
            extra_line_left = self.make_coordinates(image, extra_left_fit_average)
        if extra_line_right_fit:
            extra_right_fit_average = np.average(extra_line_right_fit, axis=0)
            extra_line_right = self.make_coordinates(image, extra_right_fit_average)

        return np.array([left_line, right_line, extra_line_left, extra_line_right])

    def region_ofinterest(self, img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            match_mask_color = (255,) * channel_count
        else:
            match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        return line_image



    def process(self, image):
        lane_image = np.copy(image)

        # ROI-------------------
        height = image.shape[0]
        width = image.shape[1]
        bottom_left = [0, 2 * height / 3]
        top_left = [0, height]
        bottom_right = [width, 2 * height / 3]
        top_right = [width, height]
        # ----------------------

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        upper_white = np.array([57, 34, 171])
        lower_white = np.array([0, 0, 131])
        mask = cv2.inRange(hsv_img, lower_white, upper_white)
        res = cv2.bitwise_and(image, image, mask=mask)
        res2 = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

        canny_image = cv2.Canny(gray_image, 100, 300)
        gaussianBlurImg = cv2.GaussianBlur(res2, (1, 1), 0)
        cropped_image = self.region_ofinterest(gaussianBlurImg,
                                            np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32), )
        #cv2.imshow('half_canny', cropped_image)
        lines = cv2.HoughLinesP(cropped_image, 1, (np.pi / 180), 50, np.array([]), minLineLength=40, maxLineGap=30)

        averaged_lines = self.average_slope_intercept(lane_image, lines)
        line_image = self.display_lines(lane_image, averaged_lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        #cv2.imshow('result', combo_image)
        return combo_image

    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)


    def detectFromImage(self, path):
        image = cv2.imread(path)
        imageScaled = self.ResizeWithAspectRatio(image, width=640)
        cv2.imshow('ORIGINAL', imageScaled)
        result = self.process(image)
        resultScaled = self.ResizeWithAspectRatio(result, width=640)
        cv2.imshow('RESULT', resultScaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detectFromVideo(self, path):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            frameScaled = self.ResizeWithAspectRatio(frame, width=640)
            cv2.imshow('ORIGINAL', frameScaled)
            frameScaled = self.process(frameScaled)
            cv2.imshow('RESULT', frameScaled)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()