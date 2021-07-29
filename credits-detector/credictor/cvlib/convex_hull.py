import cv2

import numpy as np

from credictor.cvlib.commons import lazy_property
from credictor.cvlib.image import Image


class ConvexHull(object):
    def __init__(self, coords: np.ndarray):
        self._hull = cv2.convexHull(coords)

    @lazy_property
    def area(self):
        return cv2.contourArea(self._hull)

    def draw(self, img: Image, color=(0, 0, 255), thickness=3, inplace=True) -> Image:
        if inplace:
            vis = img._array
        else:
            vis = img._array.copy()

        cv2.drawContours(vis, [self._hull], -1, color, thickness)
        return Image(vis, img._color_space)
