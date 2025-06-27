import cv2


def compare_images_orb(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 创建ORB特征提取器
    orb = cv2.ORB_create()

    # 在两张图片上检测关键点和计算特征描述子
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 创建BFMatcher匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 使用match进行特征匹配
    matches = bf.match(descriptors1, descriptors2)

    # 进行筛选，保留较好的匹配结果
    good_matches = sorted(matches, key=lambda x: x.distance)[: int(len(matches) * 0.15)]

    # 计算相似度
    similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))

    return similarity


# 示例用法
img1_path = "Pics/standard.png"
img2_path = "Pics/beforeEC.png"
similarity = compare_images_orb(img1_path, img2_path)
print("相似度：", similarity)
