import numpy as np
import cv2

from .processor import *
import matplotlib.pyplot as plt

class Extractor():
    def __init__(self) -> None:
        self.preprocess = [
            Binarize(),
            MorphologicalTransformation(),
            Sobel(),
        ]

        self.detect = Detect()
        self.out_hist = []
    
    def __call__(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if len(im.shape) > 2 else im
        self.out_hist.append(im)

        # PRE-PROCESSING
        out = im.copy()
        for op in self.preprocess:
            out = op(out)
            self.out_hist.append(out)
        
        # OPS TO DETECT THE TARGET 
        lines, intersections, points = self.detect(out)
        x = im.copy()
        x = self.extract(x, intersections)



        # VISUALIZATION
        dl = self.detect.draw_lines(im, lines)
        dp = self.detect.draw_points(im, points)
        self.out_hist.append(dl)
        self.out_hist.append(dp)

        # POST-PROCESSING
        # x = cv2.equalizeHist(x)
        # x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(x)

        # x = Binarize(100, 255, cv2.THRESH_BINARY)(x)
        self.out_hist.append(x)

        return self.out_hist

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
        imc = im.astype("uint8").copy()

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

    def plot(self, *im, title=None, save=None, cmap="gray"):
        plt.figure(dpi=600)
        plt.axis("off")
        
        im = np.concatenate(im, axis=1)
        plt.imshow(im, interpolation="nearest", cmap=cmap)

        if title is not None:
            plt.title(title)
        if save is not None:
            output_path = save + ".jpg"
            plt.savefig(output_path, bbox_inches="tight")