import numpy as np

from credictor.cvlib.image import Image


def pixel_norm_diff(frame1: Image, frame2: Image) -> float:
    return (frame1 - frame2).norm()


def hist_norm_diff(frame1: Image, frame2: Image) -> float:
    return np.linalg.norm(frame1.bw_hist() - frame2.bw_hist())
