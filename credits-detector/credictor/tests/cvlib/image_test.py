import pytest
import numpy as np

from credictor.cvlib.image import Image, ColorSpace


class TestImage:
    def test_image_sum(self):
        img1 = Image(np.full((10, 10), 253, dtype=np.uint8))
        img2 = Image(np.full((10, 10), 2, dtype=np.uint8))
        img3 = img1 + img2
        expected = np.full((10, 10), 255, dtype=np.uint8)
        assert np.array_equal(img3._array, expected)

    def test_image_subtraction(self):
        img1 = Image(np.full((10, 10), 255, dtype=np.uint8))
        img2 = Image(np.full((10, 10), 125, dtype=np.uint8))
        img3 = img1 - img2
        expected = np.full((10, 10), 130, dtype=np.uint8)
        assert np.array_equal(img3._array, expected)

    def test_color_space_detection(self):
        gray = np.full((1, 10), 100, dtype=np.uint8)
        assert Image._detect_color_space(gray) == ColorSpace.GRAY
        bgr = np.full((10, 10, 3), 100, dtype=np.uint8)
        assert Image._detect_color_space(bgr) == ColorSpace.BGR
        
    def test_adaptive_threshold_returns_error_when_receives_bgr_image(self):
        with pytest.raises(ValueError) as error:
            Image(np.full((10, 10, 3), 100, dtype=np.uint8)) \
            .adaptive_threshold(block_size = 5, constant = -25)
        assert 'Image needs to be converted to gray scale' in str(error.value)
            
        
