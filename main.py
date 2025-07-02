import cv2 as cv
import os
import json
import pytesseract
from pytesseract import Output

from iimg import Img, InfoBox


def list_path_images(path: str):
    imgs = []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")):
            imgs.append(os.path.join(path, file))
    return imgs


def deal_std(img: Img):
    return img


def deal_uut(img: Img):
    return img


def compare_images(standard: Img, uut: Img):
    standard = deal_std(standard)
    uut = deal_uut(uut)


def main():
    # uuts = list_path_images("Pics")
    # stds = list_path_images("Standards")
    # print(uuts)
    # print(stds)
    std = Img("Standards/AP23NA.png")
    uut = Img("Pics/UUT-2.jpg")

    with open("data.json", "w") as f:
        json.dump(std.data, f)
        f.write("\n")
        json.dump(uut.data, f)


if __name__ == "__main__":
    main()
