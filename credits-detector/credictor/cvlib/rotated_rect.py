import cv2
import numpy as np

from credictor.cvlib.image import Image


class RotatedRect(object):
    def __init__(self, coords: np.ndarray):
        self._rect = cv2.minAreaRect(coords.reshape(-1, 1, 2))

    @property
    def angle(self):
        return self._rect[2]

    @property
    def center(self):
        return self._rect[0]

    @property
    def height(self):
        if self.angle < -45:
            return self._rect[1][0]
        else:
            return self._rect[1][1]

    @property
    def width(self):
        if self.angle < -45:
            return self._rect[1][1]
        else:
            return self._rect[1][0]

    @property
    def area(self):
        return self.height * self.width

    @property
    def aspect_ratio(self):
        try:
            return self.width / self.height
        except ZeroDivisionError:
            return 0

    def draw(self, img: Image, color=(0, 0, 255), thickness=3, inplace=True):
        if inplace:
            vis = img
        else:
            vis = img._array.copy()
        contour = cv2.boxPoints(self._rect).astype('int0')
        cv2.drawContours(vis, [contour], -1, color, thickness)
        return Image(vis, img._color_space)
