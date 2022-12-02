import cv2
import math
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans

import imutils
import matplotlib.pyplot as plt

class Plot:
    def plot(im, title=None, save=None, cmap="gray"):
        plt.figure(dpi=600)
        plt.axis("off")

        im = np.concatenate(im, axis=1)
        plt.imshow(im, interpolation="nearest", cmap=cmap)

        if title is not None:
            plt.title(title)
        if save is not None:
            output_path = save + ".jpg"
            plt.savefig(output_path, bbox_inches="tight")

class Blur:
    def __init__(
        self, ksize=(9,9)
    ):
        self.ksize = ksize

    def __call__(self, im):
        return cv2.blur(im, self.ksize)
        

class Binarize:
    def __init__(
        self, thresh1=0, thresh2=255, mode=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    ):
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        self.mode = mode

    def __call__(self, im):
        return cv2.threshold(
            im,
            self.thresh1,
            self.thresh2,
            self.mode,
        )[1]

class MorphologicalTransformation:
    def __init__(self, shape=cv2.MORPH_ELLIPSE, ksize=(3,3), niter=22, mode="barcode"):
        self.shape = shape
        self.niter = niter
        self.ksize = ksize
        self.mode = mode

    def __call__(self, im):
        kernel = cv2.getStructuringElement(self.shape, self.ksize)
        out =  cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=self.niter)

        if self.mode.lower() == "barcode":
            out = cv2.erode(out, None, iterations=4)
            out = cv2.dilate(out, None, iterations=4)

        return out
class Sobel:
    def __init__(self, ddepth=cv2.CV_16S, ksize=3, thresh=100):
        self.ddepth = ddepth
        self.ksize = ksize
        self.thresh = thresh

        self.ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        # self.ddepth = ddepth

    def __call__(self, im):
        gradx = cv2.Sobel(im, self.ddepth, dx=1, dy=0, ksize=self.ksize)
        grady = cv2.Sobel(im, self.ddepth, dx=0, dy=1, ksize=self.ksize)

        if self.thresh is not None:
            gradx = cv2.convertScaleAbs(gradx)
            grady = cv2.convertScaleAbs(grady)
            out = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
            out = np.where(out > self.thresh, 255, 0)
        else:
            out = cv2.subtract(gradx, grady)
            out = cv2.convertScaleAbs(out)    

        return out


class Hough:
    def __init__(self, rho=1, theta=np.pi / 180, thresh=100) -> None:
        self.rho = rho
        self.theta = theta
        self.thresh = thresh

    def __call__(self, im):
        imc = im.astype("uint8").copy()
        return cv2.HoughLines(
            imc,
            self.rho,
            self.theta,
            self.thresh
        )
        

class Detect:
    def __init__(self):
        self.hough = Hough()

    def __call__(self, im):
        imc = im.astype("uint8").copy()
        lines = self.hough(im)
        intersections = self.get_intersections(imc, lines)
        points = self.find_quadrilaterals(intersections)

        return lines, intersections, points

    def get_angle(self, line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        m1 = -(np.cos(theta1) / (np.sin(theta1) + 1e-9))
        m2 = -(np.cos(theta2) / (np.sin(theta2) + 1e-9))
        return abs(math.atan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)

    def intersection(self, line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array(
            [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
        )

        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    def get_intersections(self, im, lines):

        inters = []
        pairs = combinations(range(len(lines)), 2)
        x_in_range = lambda x: 0 <= x <= im.shape[1]
        y_in_range = lambda y: 0 <= y <= im.shape[0]


        for i, j in pairs:
            line_i, line_j = lines[i][0], lines[j][0]

            if 80.0 < self.get_angle(line_i, line_j) < 100.0:
                int_point = self.intersection(line_i, line_j)

                if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]):
                    inters.append(int_point)

        return inters

    def draw_lines(self, im, lines):
        imc = im.copy()
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            n = 5000
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * (a))
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * (a))

            cv2.line(imc, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return imc

    def draw_points(self, im, points):
        imc = im.copy()
        for point in points:
            x, y = int(point[0][0]), int(point[0][1])

            cv2.circle(imc, (x, y), 5, (255, 0, 0), 5)
        return imc

    def find_quadrilaterals(self, intersections):
        X = np.array([[point[0][0], point[0][1]] for point in intersections])
        kmeans = KMeans(
            n_clusters=4, init="k-means++", max_iter=100, n_init=10, random_state=0
        ).fit(X)

        return [[center.tolist()] for center in kmeans.cluster_centers_]
