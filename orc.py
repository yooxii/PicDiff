import numpy as np
import pytesseract
from pytesseract import Output
import cv2

from PIL import Image, ImageDraw, ImageFont


def nothing(x):
    pass


def deal_pic(png):
    # 读取图像文件
    img_gray = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((10, 10), np.uint8)

    # 二值化
    binary_adaptive = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 11
    )
    cv2.imshow(f"{png}-imgGray", img_gray)
    cv2.imshow(f"{png}-binary_adaptive", binary_adaptive)
    # img_resize = cv2.resize(img_gray, None, fx=0.15, fy=0.15)
    # binary_resize = cv2.adaptiveThreshold(
    #     img_resize, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 11
    # )
    # cv2.imshow(f"{png}-imgGray", img_resize)
    # cv2.imshow(f"{png}-binary_resize", binary_resize)
    return binary_adaptive


def binary_pic(img):
    img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.resize(img_gray, None, fx=1.5, fy=1.5)
    # _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow(f"binary_img", img_gray)
    return img_gray


def dyn_bin_pic(img):
    # 创建一个黑色图像，一个窗口
    cv2.namedWindow("settings")
    # 创建一个改变颜色的轨迹栏
    cv2.createTrackbar("thresh", "settings", 0, 255, nothing)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        thresh = cv2.getTrackbarPos("thresh", "settings")
        _, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        cv2.imshow(f"binary_img", binary_img)
    return binary_img


# img_gray = cv2.imread("Pics/standard.png", cv2.IMREAD_GRAYSCALE)
# res1 = dyn_bin_pic(img_gray)

# deal_pic("Pics/UUT-2.jpg")
res1 = binary_pic("Pics/standard.png")

# 识别文字
text = pytesseract.image_to_string(res1, lang="eng", config="--psm 1")
print(text)

cv2.waitKey(0)
