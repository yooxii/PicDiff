import cv2
import numpy as np

# 1. 读取两张图片（确保尺寸一致）
image1 = cv2.imread("Pics/UUT.png")
image2 = cv2.imread("Pics/standard.png")

gray2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
_, thresh2 = cv2.threshold(gray2, 160, 255, cv2.THRESH_BINARY)
canny = cv2.Canny(gray2, 50, 100, apertureSize=3)
lines = cv2.HoughLinesP(
    canny, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=3
)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算线段的长度
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # # 只绘制长度大于某个阈值且最接近图像边缘的线段
        # if length > 100 and (
        #     x1 < 10 or x2 < 10 or x1 > image1.shape[1] - 10 or x2 > image1.shape[1] - 10
        # ):
        cv2.line(image1, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # if length > 100:
        #     cv2.line(image2, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow("image2", image1)
cv2.imshow("gray2", gray2)
cv2.imshow("thresh2", thresh2)
cv2.imshow("canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


def resize2same(img1, img2):
    # 将两张图片中较大的图片等比例缩放到较小的图片的尺寸，再裁剪掉多余的宽度
    big_img, sml_img = (img1, img2) if img1.shape[0] > img2.shape[0] else (img2, img1)
    scale = sml_img.shape[0] / big_img.shape[0]
    big_img = cv2.resize(big_img, None, fx=scale, fy=scale)
    canny = cv2.Canny(big_img, 100, 200)
    x, y, w, h = cv2.boundingRect(canny)
    cv2.imshow("canny", canny)
    big_img = big_img[y : y + h, x : x + w]
    return big_img, sml_img


# image1, image2 = resize2same(image1, image2)
#
# cv2.imshow("original1", image1)
# cv2.imshow("original2", image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 2. 检查是否成功加载图片
# if image1 is None or image2 is None:
#     print("无法加载图片，请检查路径是否正确。")
#     exit()
#
# # 3. 转换为灰度图（减少颜色影响，仅关注结构差异）
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("gray1", gray1)
# cv2.imshow("gray2", gray2)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # 二值化
# thresh1 = cv2.adaptiveThreshold(
#     gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
# )
# thresh2 = cv2.adaptiveThreshold(
#     gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7
# )
#
# cv2.imshow("thresh1", thresh1)
# cv2.imshow("thresh2", thresh2)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # 4. 计算两图之间的差异（像素级差值）
# diff = cv2.absdiff(thresh1, thresh2)
#
# # 5. 叠加阈值化后的差异图
# fix_img = cv2.addWeighted(thresh1, 0.5, thresh2, 0.5, 0)
# cv2.imshow("thresh", fix_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 6. 查找差异区域的轮廓
# contours, _ = cv2.findContours(fix_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 7. 在原始图片上画出差异区域（例如在 image1 上画框）
# for contour in contours:
#     # 忽略太小的差异区域
#     if cv2.contourArea(contour) > 5:  # 可根据需求调整面积阈值
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框
#
# # 8. 显示结果
# cv2.imshow("Differences", image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 可选：保存结果到文件
# cv2.imwrite("differences_marked.png", image2)
