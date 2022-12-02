import numpy as np
import cv2
from pyzbar.pyzbar import decode
from .processor import *

class BarcodeReader():
    def __init__(self) -> None:
        self.extract = Extractor()
        self.out = []

    def get_barcodes(self, im):
        im = self.extract(im)
        bardet = decode(im)
        
        for x in bardet:
            data = x.data.decode("UTF-8")
            if data[0] == "%":
                self.out.append({
                    "content": data,
                    "destination_postal_code": data[1:8],
                    "tracking_number": {
                        "id": data[8:22],
                        "origin_location": data[8:12],
                        "origin_parcel_number": data[12:22],
                    },
                    "service_code": data[22:25],
                    "destination_country_code": data[25:],
                })
            else:
                self.out.append(data)

        return self.out

class Extractor(Plot):
    def __init__(self) -> None:
        self.preprocess = [
            Binarize(),
            MorphologicalTransformation(mode="extract"),
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
        self.out_hist.append(x)

        return x

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract(self, im, inters):
        imc = im.astype("uint8").copy()

        pts = np.array([(x, y) for intersection in inters for x, y in intersection])
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        warped = cv2.warpPerspective(
            imc, cv2.getPerspectiveTransform(rect, dst), (maxWidth, maxHeight)
        )

        return warped