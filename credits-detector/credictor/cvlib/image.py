from enum import Enum
from typing import Optional, Any

import cv2
import numpy as np


class ColorSpace(Enum):
    BGR = 1
    GRAY = 2


class Image:
    def __init__(self, img_array: np.ndarray, color_space: Optional[ColorSpace] = None):
        self._array = img_array
        self._color_space = color_space or Image._detect_color_space(img_array)

    @property
    def array(self):
        return self._array

    @staticmethod
    def read(fname: str) -> "Image":
        img = cv2.imread(fname)
        if img is None:
            raise ValueError("Couldn't open image %s" % fname)
        return Image(img)

    @staticmethod
    def _detect_color_space(img: np.ndarray) -> ColorSpace:
        if img.ndim == 2 or img.shape[2] == 1:
            return ColorSpace.GRAY
        elif img.ndim == 3 and img.shape[2] == 3:
            return ColorSpace.BGR
        else:
            raise ValueError("Couldn't automatically detect color space for image with shape %s" % str(img.shape))

    def gray(self) -> "Image":
        if self._color_space is ColorSpace.BGR:
            gray_img = cv2.cvtColor(self._array, cv2.COLOR_BGR2GRAY)
            return Image(gray_img)
        else:
            return self
        
    def blur(self, ksize, sigma) -> "Image":
        blur_img = cv2.GaussianBlur(self._array, ksize, sigma)
        return Image(blur_img)
    
    def adaptive_threshold(self, block_size, constant, max_value = 255, method = cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type = cv2.THRESH_BINARY) -> "Image":
        if self._color_space is ColorSpace.BGR:
            raise ValueError("Image needs to be converted to gray scale")
        else:
            adapted_img = cv2.adaptiveThreshold(self._array, max_value, method, threshold_type, block_size, constant)
            return Image(adapted_img)
            
    @property
    def shape(self) -> tuple:
        return self._array.shape

    def bw_hist(self) -> np.ndarray:
        gray = self.gray()
        array = gray._array
        return np.bincount(array.ravel(), minlength=256)

    def norm(self) -> float:
        return np.linalg.norm(self._array)

    def __add__(self, other: Any) -> "Image":
        if not isinstance(other, Image):
            raise ValueError("__add__ is not defined for classes Image and %s" % str(type(other)))
        other_arr = other._array
        new = cv2.add(self._array, other_arr)
        return Image(new)

    def __sub__(self, other: Any) -> "Image":
        if not isinstance(other, Image):
            raise ValueError("__sub__ is not defined for classes Image and %s" % str(type(other)))
        other_arr = other._array
        new = cv2.subtract(self._array, other_arr)
        return Image(new)
