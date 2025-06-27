import cv2
import numpy as np


def compare_images_pixel(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 确保两张图片具有相同的尺寸
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # 计算两张图片的差异
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 将差异图像转换为二值图像
    _, threshold = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # 计算相似度
    similarity = np.mean(threshold)

    return similarity


def compare_images(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()

    # 在两张图片上检测关键点和计算特征描述子
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher()

    # 使用knnMatch进行特征匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 进行筛选，保留较好的匹配结果
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 计算相似度
    similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))

    return similarity


# 示例用法

img1_path = "afterEC.png"
img2_path = "beforeEC.png"
similarity = compare_images_pixel(img1_path, img2_path)
print("相似度：", similarity)

similarity = compare_images(img1_path, img2_path)
print("相似度：", similarity)
