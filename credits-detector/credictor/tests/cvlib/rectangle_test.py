import numpy as np

from credictor.cvlib.rectangles import Rectangle
from credictor.cvlib.image import Image

class TestRectangles:
    def test_if_detect_regions_returns_list(self):
        input_img = Image(np.full((10, 10, 3), 100, dtype=np.uint8))
        object_type = type(Rectangle(input_img).detect_regions())
        assert object_type == list
        
    def test_if_find_contours_returns_list(self):
        input_img = Image(np.full((10, 10, 3), 100, dtype=np.uint8))
        object_type = type(Rectangle(input_img).find_contours())
        assert object_type == list    
        
    def test_if_get_rectangles_returns_list(self):
        input_img = Image(np.full((10, 10, 3), 100, dtype=np.uint8))
        object_type = type(Rectangle(input_img).get_rectangles())
        assert object_type == list    
        
    def test_get_rectangles_count(self):
        img = Image.read('credictor/tests/resources/credit_example.png')
        assert Rectangle(img).count() == 26