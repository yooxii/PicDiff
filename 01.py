import cv2 as cv
import numpy as np


def nothing(x):
    pass


img_uut = cv.imread("Pics/UUT.png")
img_std = cv.imread("Pics/standard.png")
std_h, std_w = img_std.shape[:2]
uut_h = img_uut.shape[0]

img_uut = cv.resize(img_uut, None, fx=std_h / uut_h, fy=std_h / uut_h)
uut_h, uut_w = img_uut.shape[:2]


# 创建一个黑色图像，一个窗口
cv.namedWindow("settings")
# 创建一个改变颜色的轨迹栏
cv.createTrackbar("Width", "settings", 0, 50, nothing)
cv.createTrackbar("Height", "settings", 0, 50, nothing)
cv.createTrackbar("Weight", "settings", 1, 100, nothing)

while True:
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    width = cv.getTrackbarPos("Width", "settings") + std_w - uut_w
    height = cv.getTrackbarPos("Height", "settings")
    weight = cv.getTrackbarPos("Weight", "settings")
    pts1 = np.float32([[0, 0], [uut_w, 0], [0, uut_h], [uut_w, uut_h]])
    pts2 = np.float32(
        [
            [0, 0],
            [uut_w + width, 0],
            [0, uut_h + height],
            [uut_w + width, uut_h + height],
        ]
    )
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img_uut, M, (uut_w + width, uut_h + height))
    # 图像切割
    dst = dst[0:std_h, 0:std_w]
    dst = cv.addWeighted(dst, weight / 100, img_std, 1 - weight / 100, 0)
    cv.imshow("dst", dst)


cv.destroyAllWindows()
