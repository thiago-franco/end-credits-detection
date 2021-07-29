from credictor.cvlib.region import Region
from credictor.cvlib.image import Image


class TestRegion:
    def test_regions_detection(self):
        img = Image.read('credictor/tests/resources/regions3.png')
        regions = Region.detect_mser_regions(img)
        assert len(regions) == 3

    def test_circle_region(self):
        img = Image.read('credictor/tests/resources/region0.png')
        regions = Region.detect_mser_regions(img)
        assert len(regions) == 0
