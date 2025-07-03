import os
import json

import numpy as np
import cv2 as cv
import pytesseract
from pytesseract import Output


DEFAULT_WIDTH = 2000
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
        p1=(0, 0),
        p2=(0, 0),
        color="red",
        text="",
        thickness=1,
        line_type=cv.LINE_AA,
    ):
        self.p1 = p1
        self.p2 = p2
        self.color = color
        self.text = text
        self.thickness = thickness
        self.line_type = line_type


class Img:
    def __init__(self, img_path):
        self.path = img_path
        self.img = cv.imread(img_path)
        if self.img is None:
            raise ValueError("Image not found")
        self.main_img = self.img.copy()
        self.stand_img = self.img.copy()
        self.text_img = None

        self.name = os.path.basename(img_path)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.data = None

        self.img2StandardSize()

    def reread(self, pic=None):
        if isinstance(pic, str):
            self.img = cv.imread(pic)
        elif isinstance(pic, np.ndarray):
            self.img = pic
        else:
            self.img = self.main_img.copy()
        return self.img

    def img2StandardSize(self):
        """将图片裁剪至标准宽度的大小，这个宽度默认为2000像素"""
        img = self.reread()
        img = self.img2gray(img)
        img = cv.GaussianBlur(img, (25, 25), 0)
        img = cv.adaptiveThreshold(
            img, 250, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 7
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
            print("Background is white")
        cv.imwrite(f"{self.name}_background.jpg", img)

        kernel1 = np.ones((15, 15), np.uint8)
        close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel1, iterations=2)
        contours, _ = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            cropped_image = self.main_img[y : y + h, x : x + w]
        else:
            cropped_image = self.main_img
            print("No contour found")

        fxy = DEFAULT_WIDTH / cropped_image.shape[1]
        self.stand_img = cv.resize(cropped_image, None, fx=fxy, fy=fxy)
        cv.imwrite(f"{self.name}_standard_size.jpg", self.stand_img)
        return self.stand_img

    def img2gray(self, img=None):
        if img is None:
            img = self.img.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    def gray2bin_OTSU(self, img=None):
        if img is None:
            img = self.img.copy()
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        return img

    def gray2bin_adap(self, img=None):
        if img is None:
            img = self.img.copy()
        img = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 7
        )
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
        img = self.img2gray(img)
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        box1 = InfoBox(
            max_loc,
            (
                max_loc[0] + template.shape[1],
                max_loc[1] + template.shape[0],
            ),
            "red",
        )
        print(max_val)
        draw = self.draw_boxes(self.main_img.copy(), [box1])
        return draw

    def deal_logo(self):

        return self.img

    def cut_text_area(self):
        img = self.stand_img.copy()
        img = self.img2gray(img)
        img = cv.GaussianBlur(img, (25, 25), 0)
        img = cv.adaptiveThreshold(
            img, 250, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 7
        )
        # 先腐蚀再闭运算再膨胀，
        kernel = np.ones((9, 9), np.uint8)
        img = cv.erode(img, np.ones((7, 7), np.uint8))
        close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=7)
        dilation = cv.dilate(close, np.ones((25, 25), np.uint8))

        contours, _ = cv.findContours(dilation, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > (DEFAULT_WIDTH * 0.5) and (
                (DEFAULT_WIDTH * 0.2) < h < (DEFAULT_WIDTH * 0.75)
            ):
                box = InfoBox((x, y), (x + w, y + h), "red", thickness=3)
                boxes.append(box)
        draw = self.draw_boxes(self.stand_img.copy(), boxes)
        cv.imwrite(f"{self.name}_pre.jpg", dilation)
        cv.imwrite(f"{self.name}_contours.jpg", draw)
        self.text_img = self.stand_img[
            boxes[0].p1[1] : boxes[0].p2[1], boxes[0].p1[0] : boxes[0].p2[0]
        ]
        cv.imwrite(f"{self.name}_text.jpg", self.text_img)
        return self.text_img

    def deal_text(self, img, meth: str) -> dict:
        if img is None:
            raise ValueError("No Image")
        src = pytesseract.image_to_data(
            img,
            lang="eng",
            config="--psm 1 -c tessedit_write_images=true",
            output_type=Output.DICT,
        )
        data = {}
        for k in src.keys():
            data[k] = []
        # 过滤-1置信度的文字
        for i, cf in enumerate(src["conf"]):
            if cf == -1:
                continue
            for k, v in src.items():
                data[k].append(v[i])

        total_conf = 0
        average_conf = 0
        for i, cf in enumerate(data["conf"]):
            total_conf += cf
        average_conf = total_conf / len(data["conf"])
        data["total_conf"] = total_conf
        data["average_conf"] = average_conf
        data["text_num"] = len(data["text"])
        data["meth"] = meth
        return data

    def max_conf_text(self):
        _data = []

        img = self.cut_text_area()
        img = self.img2gray(img)
        _data.append(self.deal_text(img, "non"))

        adpt = self.gray2bin_adap(img)
        _data.append(self.deal_text(adpt, "adpt"))

        otsu = self.gray2bin_OTSU(img)
        _data.append(self.deal_text(otsu, "otsu"))

        cv.imwrite(f"{self.name}_adpt.jpg", adpt)
        cv.imwrite(f"{self.name}_otsu.jpg", otsu)

        for data in _data:
            if self.data is None:
                self.data = data
                continue
            if (data["average_conf"] - self.data["average_conf"] > 10):
                self.data = data
            elif data["average_conf"] > self.data["average_conf"] and (
                -5 < data["text_num"] - _data[0]["text_num"] < 5
            ):
                self.data = data
        return self.data

    def draw_boxes(self, img, boxes: list[InfoBox]):
        if img is None:
            img = self.img.copy()
        for box in boxes:
            cv.rectangle(
                img,
                box.p1,
                box.p2,
                COLOR[box.color],
                box.thickness,
                box.line_type,
            )
        return img


def test_text():
    img1 = Img("Pics/UUT-2.jpg")
    img1.max_conf_text()
    # img2 = Img("Pics/AP23NA-02.jpg")
    # img2.max_conf_text()

    with open("data1.json", "w") as f:
        json.dump(
            {
                "conf": img1.data["conf"],
                "text": img1.data["text"],
                "total_conf": img1.data["total_conf"],
                "average_conf": img1.data["average_conf"],
                "text_num": img1.data["text_num"],
                "meth": img1.data["meth"],
            },
            f,
            indent=4,
        )
    # with open("data2.json", "w") as f:
    #     json.dump(
    #         {
    #             "conf": img2.data["conf"],
    #             "text": img2.data["text"],
    #             "total_conf": img2.data["total_conf"],
    #             "average_conf": img2.data["average_conf"],
    #             "text_num": img2.data["text_num"],
    #         },
    #         f,
    #         indent=4,
    #     )


if __name__ == "__main__":
    test_text()
    # img1 = Img("Pics/AP23NA-01.jpg")
    # img2 = Img("Pics/AP27BNA-04.jpg")
    # img2.cut_text_area()
