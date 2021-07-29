from typing import List

import cv2

import numpy as np

from credictor.cvlib.commons import lazy_property
from credictor.cvlib.convex_hull import ConvexHull
from credictor.cvlib.image import Image
from credictor.cvlib.rotated_rect import RotatedRect


class Region(object):
    def __init__(self, coords: np.ndarray):
        self._coords = coords

    @property
    def coords(self) -> np.ndarray:
        return self.coords

    @lazy_property
    def rotated_rect(self) -> RotatedRect:
        return RotatedRect(self._coords)

    @lazy_property
    def convex_hull(self) -> ConvexHull:
        return ConvexHull(self._coords)

    @lazy_property
    def solidity(self) -> float:
        convex_hull_area = self.convex_hull.area
        rotated_rect_area = self.rotated_rect.area
        return convex_hull_area / rotated_rect_area

    def draw(self, img: Image, color=(0, 0, 255), thickness=1, inplace=True):
        if inplace:
            vis = img._array
        else:
            vis = img._array.copy()
        cv2.drawContours(vis, [self._coords], -1, color, thickness)
        return Image(vis, img._color_space)

    @staticmethod
    def detect_mser_regions(img: Image) -> List['Region']:
        gray_img = img.gray()._array
        mser = cv2.MSER_create()
        regions = mser.detectRegions(gray_img)
        return [Region(r) for r in regions[0]]
