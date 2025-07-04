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
    # Load standard and test images
    std = Img("Standards/AP23NA.png")
    uut = Img("Pics/UUT-2.jpg")

    # Compare images and get differences
    diff_image, diff_data = std.compare_with(uut)
    
    # Save results
    cv.imwrite("diff_result.jpg", diff_image)
    
    with open("diff_data.json", "w") as f:
        json.dump(diff_data, f, indent=4)
    
    print("Comparison completed:")
    print(f"- Differences saved to diff_result.jpg")
    print(f"- Difference data saved to diff_data.json")


if __name__ == "__main__":
    main()
