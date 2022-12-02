from utils.scanner import Extractor, BarcodeReader

import matplotlib.pyplot as plt

import cv2
import numpy as np
import time


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)

args = parser.parse_args()
filename = args.image

im = cv2.imread(filename)

bar = BarcodeReader()

out = bar.get_barcodes(im)
print(out)

