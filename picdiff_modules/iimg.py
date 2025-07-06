from typing import Optional
import cv2 as cv
import numpy as np
from picdiff_modules.image_processor import ImageProcessor
from picdiff_modules.ocr_engine import OCREngine
from picdiff_modules.diff_analyzer import DiffAnalyzer
from picdiff_modules.image_utils import InfoBox, draw_boxes


class Img:
    def __init__(self, img_path: str, logo_path: str = ""):
        """初始化图像对象

        参数:
            img_path: 图像文件路径
            logo_path: logo图像文件路径(可选)
        """
        self.img_path = img_path
        self.logo_path = logo_path
        self.name = img_path.split("/")[-1].split(".")[0]  # 从路径提取文件名
        self.img = cv.imread(img_path, cv.IMREAD_COLOR)
        self.output_img = None  # 输出图像

        if self.img is None:
            raise ValueError(f"加载图像失败: {img_path}")

        # 确保图像是3通道BGR格式
        if len(self.img.shape) == 2:
            self.img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        elif self.img.shape[2] == 4:
            self.img = cv.cvtColor(self.img, cv.COLOR_BGRA2BGR)

        self.text_img = None  # 存储带文本标记的图像
        self.data = {}  # 存储OCR识别结果

        self.processor = ImageProcessor()  # 图像处理器
        self.ocr = OCREngine()  # OCR引擎
        self.differ = DiffAnalyzer()  # 差异分析器

    def resize_to_standard(self) -> None:
        """将图像调整为标准尺寸"""
        self.img = self.processor.resize_to_standard(self.img)
        self.output_img = self.img.copy()

    def preprocess_image(self) -> None:
        """应用自适应预处理"""
        self.img = self.processor.preprocess(self.img)

    def extract_text(self, method: str = "auto") -> dict:
        """使用指定方法从图像中提取文本

        参数:
            method: 识别方法('auto'自动选择最佳方法)
        返回:
            包含识别结果的字典
        """
        if method == "auto":
            self.data = self.ocr.get_best_ocr_result(self.img)
        else:
            self.data = self.ocr.process_image(self.img, method)

        # 在图像上绘制文本区域框
        self.text_img = draw_boxes(
            self.img.copy(),
            [
                InfoBox(
                    x=box[0], y=box[1], w=box[2], h=box[3], color="green"
                )  # 绿色框(0,255,0)
                for box in zip(
                    self.data["left"],
                    self.data["top"],
                    self.data["width"],
                    self.data["height"],
                )
            ],
        )
        return self.data

    def compare_with(self, other: "Img") -> tuple[dict, list]:
        """与另一幅图像比较logo和文本内容

        参数:
            other: 要比较的另一幅图像对象
        返回:
            包含比较结果的字典
        """
        # 首先比较logo(如果提供了logo路径)
        logo_result = {}
        if self.logo_path and other.logo_path:
            logo_img = cv.imread(self.logo_path)
            other_logo = cv.imread(other.logo_path)
            if logo_img is not None and other_logo is not None:
                logo_result = self.differ.compare_logo(self.img, other_logo)

        # 然后比较文本内容
        self.extract_text()
        other.extract_text()

        text_diff, diff_boxes = self.differ.compare_text_content(self.data, other.data)

        return {
            "logo_comparison": logo_result,
            "text_comparison": text_diff,
            "original_text": {
                "standard": self.data["text"],
                "tested": other.data["text"],
            },
        }, diff_boxes
