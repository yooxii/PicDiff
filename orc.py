import numpy as np
import pytesseract
from pytesseract import Output
import cv2

from PIL import Image, ImageDraw, ImageFont


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
    return binary_adaptive


# deal_pic("Pics/UUT.png")
# res1 = deal_pic("Pics/standard.png")

# 识别文字
text = pytesseract.image_to_string("Pics/UUT.png", lang="eng")
print(text)

cv2.waitKey(0)
