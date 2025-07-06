import cv2 as cv
import numpy as np

from iimg import Img


def diff_two_images(uut: Img, stand: Img):
    uutimg = uut.cut_text_area()
    standimg = stand.cut_text_area()

        