import numpy as np
import cv2

from .processor import *

class Extractor():
    def __init__(self) -> None:
        self.preprocess = [
            Binarize(),
            MorphologicalTransformation(),
            Sobel(),
        ]

        self.hough = Hough()
    
    def __call__(self, im):
        out_hist = []

        out = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if len(im.shape) > 2 else im
        out_hist.append(out)

        x = out.copy()
        for op in self.preprocess:
            out = op(out)
            out_hist.append(out)
        
        lines = self.hough(out)
        inter = self.hough.get_intersections(x, lines)
        points = self.hough.find_quadrilaterals(inter)

        drawn_lines = self.hough.draw_lines(x, lines)
        drawn_points = self.hough.draw_points(x, points)
        out_hist.append(drawn_lines)
        out_hist.append(drawn_points)

        out = self.extract(x, inter)
        out_hist.append(out)

        return (out, out_hist)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract(self, im, inters):
        imc = im.copy()

        pts = np.array([
            (x, y)
            for intersection in inters
            for x, y in intersection
        ])
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
                [0, 0],                         
                [maxWidth - 1, 0],              
                [maxWidth - 1, maxHeight - 1],  
                [0, maxHeight - 1]
            ], dtype = "float32")            

        warped = cv2.warpPerspective(
                imc,
                cv2.getPerspectiveTransform(rect, dst),
                (maxWidth, maxHeight))

        return warped
