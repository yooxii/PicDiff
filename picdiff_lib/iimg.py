import os
import json
import difflib

import numpy as np
import cv2 as cv

from .draw_utils import *
from .ocr_utils import *

# from picdiff_lib.image_processor import *


DEFAULT_BIG_WIDTH = 2000
DEFAULT_SMALL_WIDTH = 400


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

        self.name = os.path.basename(img_path).split(".")[0]
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

        img = cv.adaptiveThreshold(
            img, 190, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 7
        )

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
        cv.imwrite(f"{self.name}_size_colse.jpg", close)
        cv.imwrite(f"{self.name}_size_cuted.jpg", self.stand_img)
        return self.stand_img

    def match_template(self, template_path):
        """在图片中匹配传入的模板图片，传入模板图片必须小于图片本身大小

        Args:
            template_path (str): 传入模板图片路径

        Returns:
            (MatLike | ndarray | Any): 匹配结果
        """
        template = cv.imread(template_path, 0)
        img = self.stand_img.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        print(max_val)
        if max_val < 0.7:
            return None
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
        if logo_path == "":
            # print("No logo path provided")
            self.no_logo_img = self.stand_img.copy()
            return self.no_logo_img
        box = self.match_template(logo_path)
        if box is None:
            # print("No logo found")
            self.no_logo_img = self.stand_img.copy()
            return self.no_logo_img
        self.logo_box = box
        # 计算logo 周围区域的主要颜色
        x, y, w, h = box.x - 20, box.y - 50, box.w + 50, box.h + 50
        roi = self.stand_img.copy()[y : y + h, x : x + w]
        Z = roi.reshape((-1, 3))
        # 转换为np.float32
        Z = np.float32(Z)
        # 设定K均值聚类标准
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret, label, center = cv.kmeans(
            Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
        )
        # 将中心点转换回uint8类型
        center = np.uint8(center)
        # 获取最高频颜色
        main_color = center[np.argmax(np.bincount(label.flatten()))]
        box.color = tuple(main_color.tolist())
        box.thickness = -1

        self.no_logo_img = draw_boxes(self.stand_img.copy(), [box])
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
        """裁剪出文本区域，方便后续的OCR识别，有两种方法：cv和ocr

        Args:
            method (str, optional):
        cv方法：通过预设的算法，检测出文本区域。\n
        ocr方法：先对图片进行预先的OCR识别，定位出小的文本区域，再合并为完整的文本区域。\n
        默认值 "cv".

        Returns:
            np.ndarray: 裁剪出的文本区域灰度图像。
        """
        if method == "ocr":
            dilation = self.cut_text_area_ocr()
        else:
            img = self.deal_logo()
            # img = self.process.preprocess(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if self.default_width > 1000:
                dilation = self.cut_text_area_cv1(img)
            else:
                dilation = self.cut_text_area_cv2(img)

        contours, _ = cv.findContours(dilation, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > (self.default_width * 0.4) and (
                (self.default_width * 0.1) < h < (self.default_width * 0.85)
            ):
                box = InfoBox(x, y, w, h, color="red", thickness=3)
                boxes.append(box)
        draw = draw_boxes(self.no_logo_img.copy(), boxes)
        cv.imwrite(f"{self.name}_{method}_text_pre.jpg", dilation)
        cv.imwrite(f"{self.name}_{method}_text_contours.jpg", draw)

        if len(boxes) == 0:
            print("No text area found")
            self.text_img = self.no_logo_img.copy()
            self.text_img = cv.cvtColor(self.text_img, cv.COLOR_BGR2GRAY)
            if self.background_color == 1:
                self.text_img = cv.bitwise_not(self.text_img)
            return self.text_img

        self.text_img = self.no_logo_img[
            boxes[0].p1[1] : boxes[0].p2[1], boxes[0].p1[0] : boxes[0].p2[0]
        ]
        cv.imwrite(f"{self.name}_text.jpg", self.text_img)
        self.text_img = cv.cvtColor(self.text_img, cv.COLOR_BGR2GRAY)
        if self.background_color == 1:
            self.text_img = cv.bitwise_not(self.text_img)
        return self.text_img

    def cut_text_area_cv1(self, img):
        img = cv.GaussianBlur(img, (25, 25), 0)  # 高斯模糊，去除小噪声
        if self.background_color == 1:  # 背景为白色则取反
            img = cv.bitwise_not(img)
        img = cv.adaptiveThreshold(
            img, 250, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 7
        )
        # 先腐蚀再闭运算再膨胀
        kernel = np.ones((9, 9), np.uint8)
        img = cv.erode(img, np.ones((7, 7), np.uint8))
        close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=7)
        dilation = cv.dilate(close, np.ones((25, 25), np.uint8))
        return dilation

    def cut_text_area_cv2(self, img):
        img = cv.GaussianBlur(img, (3, 3), 0)  # 高斯模糊，去除小噪声
        if self.background_color == 1:  # 背景为白色则取反
            img = cv.bitwise_not(img)
        _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        # 先腐蚀再闭运算再膨胀
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

    # 裁剪文本区域
    uut.cut_text_area(method="cv")
    stand.cut_text_area(method="cv")
    # 识别文本
    data_uut = max_conf_text(img=uut.text_img.copy(), name=uut.name)
    draw_all_text(
        uut.text_img.copy(), name=uut.name, data=data_uut, color="blue", thickness=1
    )
    data_std = max_conf_text(stand.text_img.copy(), stand.name)
    draw_all_text(
        stand.text_img.copy(), name=stand.name, data=data_std, color="blue", thickness=1
    )

    # 计算差异
    htmldiff = difflib.HtmlDiff()
    hdiff = htmldiff.make_table(data_uut["text"], data_std["text"], context=False)
    differ = difflib.Differ()
    diff = list(differ.compare(data_uut["text"], data_std["text"]))

    # 计算差异区域的坐标
    diff_boxes = []
    img_ps = """
<li>如果两张图片的待确认区域OCR识别的置信度都低于60（最高100），则可能是误差，标为白色</li>
<li>如果标样图片的置信度低于60，标为青色</li>
<li>如果单体图片的置信度低于60，标为蓝色</li>
<li>如果两张图片的待确认区域置信度都高于60，则可能是<b>文本差异</b>，标为红色</li>"""
    for d_i, dif in enumerate(diff):
        if dif[0] == " ":
            continue
        if dif[0] == "-":
            i = data_uut["text"].index(dif[2:])
            try:
                j = data_std["text"].index(diff[d_i + 1][2:])
            except Exception:
                j = -1
            x = data_uut["left"][i]
            y = data_uut["top"][i]
            w = data_uut["width"][i]
            h = data_uut["height"][i]
            if j == -1:
                color = "white"
            elif data_std["conf"][j] < 60 and data_uut["conf"][i] < 60:
                color = "white"
            elif data_std["conf"][j] < 60 and data_uut["conf"][i] >= 60:
                color = "cyan"
            elif data_std["conf"][j] >= 60 and data_uut["conf"][i] < 60:
                color = "blue"
            else:
                color = "red"

            diff_boxes.append(
                InfoBox(
                    x,
                    y,
                    w,
                    h,
                    color=color,
                    text=f"{i+1}",
                    thickness=1 if uut.default_width < DEFAULT_BIG_WIDTH else 3,
                )
            )

    # 在图像中标注差异
    uut_img = cv.cvtColor(uut.text_img.copy(), cv.COLOR_GRAY2BGR)
    diff_image = draw_boxes(uut_img, diff_boxes)

    return (
        diff_image,
        {
            "diff": diff,
            "uut": {"text": data_uut["text"], "conf": data_uut["conf"]},
            "std": {"text": data_std["text"], "conf": data_std["conf"]},
        },
        hdiff,
        img_ps,
    )


def list_path_images(path: str):
    imgs = []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")):
            imgs.append(os.path.join(path, file))
    return imgs


def generate_html(
    image_path: str,
    hdiff: dict,
    report_name: str,
    img_ps: str,
    output_dir=".",
) -> str:
    """生成HTML报告

    参数:
        image_path: 标注图像路径
        hdiff: 差异分析结果
        report_name: 报告名称
        output_dir: 输出目录
    返回:
        生成的HTML文件路径
    """

    # 生成HTML
    html_path = os.path.join(output_dir, f"{report_name}-diff-report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            f"""
<!DOCTYPE html>
<html>
<head>
    <title>PicDiff Report - {report_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ display: flex; }}
        .text-comparison {{ flex: 1; padding: 20px; }}
        .image-container {{ flex: 1; padding: 20px; text-align: center; }}
        img {{ max-width: 500px; height: auto; }}
        .error {{ margin: 5px 0; padding: 5px; }}
        table.diff {{font-family:Courier; border:medium;}}
        .diff_header {{background-color:#e0e0e0}}
        td.diff_header {{text-align:right}}
        .diff_next {{background-color:#c0c0c0}}
        .diff_add {{background-color:#aaffaa}}
        .diff_chg {{background-color:#ffff77}}
        .diff_sub {{background-color:#ffaaaa}}
        .diff-span {{text-align:left;font-size:14px;}}
    </style>
</head>
<body>
    <h1>PicDiff Report - {report_name}</h1>
    <div class="container">
        <div class="image-container">
            <h2>{report_name} 差异图</h2>
            <img src="{image_path}" alt="差异图"><br>
            <ul class="diff-span">{img_ps}</ul>
        </div>
        <div class="text-comparison">
            <h2>文本比对结果</h2>
            <p style="font-size:14px">图片数字对应单体列</p>
            {hdiff}
        </div>
    </div>
</body>
</html>
            """.replace(
                "<tbody>",
                """<thead><tr class="diff_header">
                <th colspan="3">单体</th>
                <th colspan="3">标样</th>
            </tr>
        </thead><tbody>""",
            )
        )

    return html_path
