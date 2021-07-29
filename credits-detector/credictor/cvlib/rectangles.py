import cv2
import numpy as np

from credictor.cvlib.image import Image

MSER_DELTA = 4
MSER_MIN_AREA = 10
MSER_MAX_AREA = 8000
MSER_MAX_VARIATION = 0.8
MSER_MIN_DIVERSITY = 0.2
MSER_MAX_EVOLUTION = 200
MSER_AREA_THRESHOLD = 1.01
MSER_MIN_MARGIN = 0.003
MSER_EDGE_BLUR_SIZE = 5

class Rectangle(object):
    def __init__(self, frame: Image):
        self._img = frame
        self._coords = frame._array
        self._height, self._width, self._depth = self._coords.shape
        self._mser = cv2.MSER_create(MSER_DELTA, MSER_MIN_AREA, MSER_MAX_AREA, MSER_MAX_VARIATION, MSER_MIN_DIVERSITY, MSER_MAX_EVOLUTION, MSER_AREA_THRESHOLD, MSER_MIN_MARGIN, MSER_EDGE_BLUR_SIZE)
        self._transformed_img = self.transform_image()
        
    def transform_image(self) -> "Image":
        return self._img.gray() \
               .blur(ksize = (3,3), sigma = 0) \
               .adaptive_threshold(block_size = 5, constant = -25)
        
    def detect_regions(self) -> list:
        contours, bboxes = self._mser.detectRegions(self._transformed_img.array)
        
        regions = []
        for box in bboxes:
            [x,y,w,h] = box
            
            if (w < 2) or (h < 2):
                continue
                
            if ((float(w*h) / (self._width * self._height)) > 0.005) or (float(w) / h > 1):
                continue
                
            regions.append(box)
            
        return regions    
    
    def find_contours(self, xscale = 12, yscale = 0) -> list:
        regions = self.detect_regions()
        mask = np.zeros((self._height,self._width, 1), np.uint8)
        for region in regions:
            [x,y,w,h] = region
            cv2.rectangle(mask, (x-xscale, y-yscale), (x+w+xscale, y+h+yscale), (255,255,255), cv2.FILLED)
        
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours[1]
    
    def get_rectangles(self) -> list:
        contours = self.find_contours()
        filtered_contours = list(filter(self.filter_contours, contours))
        rectangles = [cv2.boundingRect(contour) for contour in filtered_contours]
        return list(filter(self.filter_rectangles, rectangles))
    
    def filter_contours(self, contour):
        perimeter = cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(poly) > 8:
            return False
        else:
            return True
    
    def filter_rectangles(self, rect):
        [x,y,w,h] = rect
        product = float(w * h)
        ratio = float(w) / h
        
        if (product / (self._width * self._height)) < 0.006:
            if ((product / (self._width * self._height)) < 0.0018) or (ratio < 2.5):
                return False
            else:
                return True
        else:
            if ratio < 1.8:
                return False
            else:
                return True
    
    def count(self):
        rectangles = self.get_rectangles()
        return len(rectangles)
            