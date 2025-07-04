import os
import json
import difflib
import re

import numpy as np
import cv2 as cv
import pytesseract
from pytesseract import Output


DEFAULT_BIG_WIDTH = 2000
DEFAULT_SMALL_WIDTH = 400

COLOR = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


class InfoBox:
    def __init__(
        self,
        x: int = None,
        y: int = None,
        w: int = None,
        h: int = None,
        p1: tuple[int, int] = (0, 0),
        p2: tuple[int, int] = (0, 0),
        color="red",
        text="",
        thickness=1,
        line_type=cv.LINE_AA,
    ):
        if None in [x, y, w, h] and p1 != p2:
            x, y, w, h = cv.boundingRect(np.array([p1, p2]))
        if p1 == p2:
            p1 = (x, y)
            p2 = (x + w, y + h)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.p1 = p1
        self.p2 = p2
        self.color = color
        self.text = text
        self.thickness = thickness
        self.line_type = line_type


def deal_text(img, meth: str) -> dict:
    if img is None:
        raise ValueError("No Image")
    src = pytesseract.image_to_data(
        img,
        lang="eng",
        config="--psm 1",
        output_type=Output.DICT,
    )
    data = {}
    for k in src.keys():
        data[k] = []
    # 过滤-1置信度的文字
    # 过滤非白名单的文字
    whitelist = ["A-Z0-9", "():\\./\-"]
    for i, (cf, text) in enumerate(zip(src["conf"], src["text"])):
        if cf == -1:
            continue
        text = re.sub(r"[^" + "".join(whitelist) + "]", "", text)
        if len(text) == 0:
            continue
        src["text"][i] = text
        for k, v in src.items():
            data[k].append(v[i])

    total_conf = 0
    for i, cf in enumerate(data["conf"]):
        total_conf += cf
    average_conf = total_conf / len(data["conf"]) if len(data["conf"]) > 0 else 0
    data["total_conf"] = total_conf
    data["average_conf"] = average_conf
    data["text_num"] = len(data["text"])
    data["meth"] = meth
    return data


def max_conf_text(img, name):
    _data = []

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _data.append(deal_text(img, "non"))

    adpt = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 7
    )
    _data.append(deal_text(adpt, "adpt"))

    otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    _data.append(deal_text(otsu, "otsu"))

    cv.imwrite(f"{name}_textadpt.jpg", adpt)
    cv.imwrite(f"{name}_textotsu.jpg", otsu)

    ret = _data[0]
    for data in _data:
        if data["average_conf"] - ret["average_conf"] > 10:
            ret = data
        elif data["average_conf"] > ret["average_conf"] and (
            -5 < data["text_num"] - _data[0]["text_num"] < 5
        ):
            ret = data
    return ret


def draw_boxes(img, boxes: list[InfoBox]):
    if img is None:
        raise ValueError("No Image")
    for box in boxes:
        cv.rectangle(
            img,
            box.p1,
            box.p2,
            COLOR[box.color] if isinstance(box.color, str) else box.color,
            box.thickness,
            box.line_type,
        )
    return img


def draw_all_text(img, name, data, color="red", thickness=1):
    boxex = []
    for i, text in enumerate(data["text"]):
        if len(text) == 0:
            continue
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        box = InfoBox(x, y, w, h, color=color, thickness=thickness)
        boxex.append(box)
    draw = draw_boxes(img, boxex)
    cv.imwrite(f"{name}_alltext.jpg", draw)
    return draw


class Img:
    def __init__(self, img_path, logo_path=""):
        self.path = img_path
        self.img = cv.imread(img_path)
        if self.img is None:
            raise ValueError("Image not found")
        self.main_img = self.img.copy()
        self.stand_img = None
        self.no_logo_img = self.img.copy()
        self.text_img = None

        self.logo_path = logo_path
        self.background_color = 0

        self.name = os.path.basename(img_path)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.default_width = (
            DEFAULT_BIG_WIDTH
            if self.img.shape[1] > DEFAULT_BIG_WIDTH
            else DEFAULT_SMALL_WIDTH
        )
        self.data = None

        self.img2standard_size()

    def reread(self, pic=None):
        if isinstance(pic, str):
            self.img = cv.imread(pic)
        elif isinstance(pic, np.ndarray):
            self.img = pic
        else:
            self.img = self.main_img.copy()
        return self.img

    def img2standard_size(self):
        """将图片裁剪至标准宽度的大小，这个宽度默认为2000像素"""
        img = self.reread()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (25, 25), 0)
        img = cv.adaptiveThreshold(
            img, 190, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 7
        )
        # 判断图片是否是白色背景，如果是白色背景则取反
        # 获取图像的尺寸
        height, width = img.shape

        # 定义角部区域的大小
        corner_size = 50

        # 提取四个角部区域的像素值
        top_left = img[0:corner_size, 0:corner_size]
        top_right = img[0:corner_size, width - corner_size : width]
        bottom_left = img[height - corner_size : height, 0:corner_size]
        bottom_right = img[height - corner_size : height, width - corner_size : width]

        # 计算四个角部区域的平均像素值
        mean_top_left = np.mean(top_left)
        mean_top_right = np.mean(top_right)
        mean_bottom_left = np.mean(bottom_left)
        mean_bottom_right = np.mean(bottom_right)

        # 计算四个角部区域的平均值
        mean_corners = (
            mean_top_left + mean_top_right + mean_bottom_left + mean_bottom_right
        ) / 4

        # 判断背景颜色
        if mean_corners > 180:
            img = cv.bitwise_not(img)
            self.background_color = 1
            print("Background is white")
        cv.imwrite(f"{self.name}_background.jpg", img)

        kernel1 = np.ones((15, 15), np.uint8)
        close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel1, iterations=9)
        contours, _ = cv.findContours(close, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            cropped_image = self.main_img[y : y + h, x : x + w]
        else:
            cropped_image = self.main_img
            print("No contour found")

        fxy = self.default_width / cropped_image.shape[1]
        self.stand_img = cv.resize(cropped_image, None, fx=fxy, fy=fxy)
        cv.imwrite(f"{self.name}_std_colse.jpg", close)
        cv.imwrite(f"{self.name}_standard_size.jpg", self.stand_img)
        return self.stand_img

    def img2gray(self, img=None):
        if img is None:
            img = self.img.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    def match_template(self, template_path):
        """在图片中匹配传入的模板图片，传入模板图片必须小于图片本身大小

        Args:
            template_path (str): 传入模板图片路径

        Returns:
            (MatLike | ndarray | Any): 匹配结果
        """
        template = cv.imread(template_path, 0)
        img = self.main_img.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        print(max_val)
        box = InfoBox(
            p1=max_loc,
            p2=(
                max_loc[0] + template.shape[1],
                max_loc[1] + template.shape[0],
            ),
            color="red",
        )
        return box

    def deal_logo(self, logo_path=None):
        if logo_path is None:
            logo_path = self.logo_path
        box = self.match_template(logo_path)
        self.logo_box = box
        # 计算logo 周围区域的主要颜色
        x, y, w, h = box.x - 20, box.y - 50, box.w + 50, box.h + 50
        roi = self.main_img.copy()[y : y + h, x : x + w]
        Z = roi.reshape((-1, 3))
        # 转换为np.float32
        Z = np.float32(Z)
        # 设定K均值聚类标准
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8  # 比如我们希望将颜色数量减少到8种
        ret, label, center = cv.kmeans(
            Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
        )
        # 将中心点转换回uint8类型
        center = np.uint8(center)
        # 获取最高频颜色
        main_color = center[np.argmax(np.bincount(label.flatten()))]
        box.color = tuple(main_color.tolist())
        box.thickness = -1

        self.no_logo_img = draw_boxes(self.main_img.copy(), [box])
        return self.no_logo_img

    def cut_text_area_ocr(self):
        img = self.main_img.copy()
        data = max_conf_text(img, self.name)
        img2 = np.zeros(img.shape, np.uint8)
        draw = draw_all_text(img2, self.name, data, color="white", thickness=-1)
        draw = cv.cvtColor(draw, cv.COLOR_BGR2GRAY)
        if self.default_width < 1000:
            kernel = np.ones((5, 5), np.uint8)
            close = cv.morphologyEx(draw, cv.MORPH_CLOSE, kernel, iterations=3)
            dilation = cv.dilate(close, np.ones((9, 9), np.uint8))
        else:
            kernel = np.ones((9, 9), np.uint8)
            close = cv.morphologyEx(draw, cv.MORPH_CLOSE, kernel, iterations=7)
            dilation = cv.dilate(close, np.ones((25, 25), np.uint8))

        return dilation

    def cut_text_area(self, method="cv"):
        if method == "cv":
            img = self.deal_logo()
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if self.default_width > 1000:
                dilation = self.cut_text_area_cv1(img)
            else:
                dilation = self.cut_text_area_cv2(img)
        elif method == "ocr":
            dilation = self.cut_text_area_ocr()
        else:
            raise ValueError("Invalid method")

        contours, _ = cv.findContours(dilation, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > (self.default_width * 0.5) and (
                (self.default_width * 0.2) < h < (self.default_width * 0.75)
            ):
                box = InfoBox(x, y, w, h, color="red", thickness=3)
                boxes.append(box)
        draw = draw_boxes(self.no_logo_img.copy(), boxes)
        cv.imwrite(f"{self.name}_{method}_cutpre.jpg", dilation)
        cv.imwrite(f"{self.name}_{method}_cutcontours.jpg", draw)

        if len(boxes) == 0:
            print("No text area found")
            self.text_img = self.no_logo_img
            return self.no_logo_img

        self.text_img = self.no_logo_img[
            boxes[0].p1[1] : boxes[0].p2[1], boxes[0].p1[0] : boxes[0].p2[0]
        ]
        cv.imwrite(f"{self.name}_text.jpg", self.text_img)
        return self.text_img

    def cut_text_area_cv1(self, img):
        img = cv.GaussianBlur(img, (25, 25), 0)
        img = cv.adaptiveThreshold(
            img, 250, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 7
        )
        # 先腐蚀再闭运算再膨胀，
        kernel = np.ones((9, 9), np.uint8)
        img = cv.erode(img, np.ones((7, 7), np.uint8))
        close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=7)
        dilation = cv.dilate(close, np.ones((25, 25), np.uint8))
        return dilation

    def cut_text_area_cv2(self, img):
        img = cv.GaussianBlur(img, (3, 3), 0)
        _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        # 先腐蚀再闭运算再膨胀，
        kernel = np.ones((3, 3), np.uint8)
        img = cv.erode(img, np.ones((3, 3), np.uint8))
        close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=7)
        close = cv.erode(close, np.ones((3, 3), np.uint8))
        dilation = cv.dilate(close, np.ones((7, 7), np.uint8), iterations=3)
        return dilation


def compare_text(uut: Img, stand: Img):
    """比较从两张图片中提取出的文本的差异

    Args:
        stand (Img): 标样图片
        uut (Img): 单体图片

    Returns:
        tuple: (diff_image, diff_data) diff_image为两张图片的差异图，diff_data为两张图片的差异数据
    """
    # Get OCR results for both images
    uut.cut_text_area(method="ocr")
    stand.cut_text_area(method="ocr")
    data_uut = max_conf_text(uut.text_img.copy(), uut.name)
    data_std = max_conf_text(stand.text_img.copy(), stand.name)

    # Find text differences
    htmldiff = difflib.HtmlDiff()
    hdiff = htmldiff.make_file(data_uut["text"], data_std["text"], context=True)
    with open(f"{uut.name}_diff.html", "w") as f:
        f.write(hdiff)
    differ = difflib.Differ()
    diff = list(differ.compare(data_uut["text"], data_std["text"]))

    # Find changed words and their positions
    diff_boxes = []
    for dif in diff:
        if dif[0] == " ":
            continue
        if dif[0] == "-":
            i = data_uut["text"].index(dif[2:])
            # Get position from OCR data
            x = data_uut["left"][i]
            y = data_uut["top"][i]
            w = data_uut["width"][i]
            h = data_uut["height"][i]

            # Create annotation box
            diff_boxes.append(
                InfoBox(
                    x,
                    y,
                    w,
                    h,
                    color="red",
                    text=f"{dif[2:]}",
                    thickness=1 if uut.default_width < DEFAULT_BIG_WIDTH else 3,
                )
            )

    # Draw differences on image
    diff_image = draw_boxes(uut.text_img.copy(), diff_boxes)

    return diff_image, {
        "diff": diff,
        "uut_text": data_uut["text"],
        "std_text": data_std["text"],
    }


def list_path_images(path: str):
    imgs = []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")):
            imgs.append(os.path.join(path, file))
    return imgs


def test_size():
    uuts = list_path_images("Pics")
    imgs = []
    for uut in uuts:
        img = Img(uut, logo_path="Standards/AP23BNA-logo.png")
        imgs.append(img)
        img.cut_text_area(method="cv")


def test_text():
    uut = Img(
        "Pics/AP37-03.jpg",
        logo_path="Standards/AP23BNA-logo.png",
    )
    # img2 = Img(
    #     "Pics/AP23NA-02.jpg",
    #     logo_path="Standards/AP23BNA-logo.png",
    # )
    # std = Img(
    #     "Pics/AP37-02.jpg",
    #     logo_path="Standards/AP23BNA-logo.png",
    # )
    # img2.cut_text_area(method="ocr")
    # with open("data2.json", "w") as f:
    #     json.dump(img2.data, f, indent=4)

    # diff_image, diff_data = compare_text(uut, std)
    # cv.imwrite(f"{uut.name}diff.jpg", diff_image)
    # with open(f"{uut.name}diff.json", "w") as f:
    #     json.dump(diff_data, f, indent=4)


if __name__ == "__main__":
    # test_text()
    test_size()
