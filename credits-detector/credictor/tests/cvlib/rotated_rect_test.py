import numpy as np

from credictor.cvlib.rotated_rect import RotatedRect


class TestRotatedRect:
    def test_unit_area(self):
        rect = RotatedRect(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]))
        assert rect.area == 1

    def test_big_area(self):
        rect = RotatedRect(np.array([[-10, -10], [1, 1], [-10, 1], [1, -10]]))
        assert rect.area == 121

    def test_square_aspect_ratio(self):
        rect = RotatedRect(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]))
        assert rect.aspect_ratio == 1

    def test_portrait_aspect_ratio(self):
        rect = RotatedRect(np.array([[0, 0], [0, 20], [10, 0], [10, 20]]))
        assert rect.aspect_ratio == 0.5

    def test_landscape_aspect_ratio(self):
        rect = RotatedRect(np.array([[0, 0], [20, 0], [0, 10], [20, 10]]))
        assert rect.aspect_ratio == 2
