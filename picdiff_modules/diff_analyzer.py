import difflib
import cv2 as cv
import numpy as np
from typing import Dict, Tuple
from .image_utils import InfoBox


class DiffAnalyzer:
    def __init__(self):
        """初始化差异分析器"""
        self.diff_methods = ["text", "visual", "both"]  # 支持的比较方法

    def compare_text(self, data1: dict, data2: dict) -> Tuple[str, list]:
        """比较两个文本列表并返回HTML差异和原始差异

        参数:
            data1: 第一个OCR结果文本列表
            data2: 第二个OCR结果文本列表
        返回:
            (HTML格式差异, 原始差异列表)
        """
        htmldiff = difflib.HtmlDiff()
        html_diff = htmldiff.make_table(
            data1["text"], data2["text"], context=True
        )  # 生成HTML差异

        differ = difflib.Differ()
        raw_diff = list(differ.compare(data1["text"], data2["text"]))  # 生成原始差异

        return html_diff, raw_diff

    def compare_logo(self, img: np.ndarray, logo: np.ndarray) -> dict:
        """使用模板匹配比较logo

        参数:
            img: 输入图像
            logo: logo模板图像
        返回:
            包含相似度和位置信息的字典
        """
        # 转换为灰度图像
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        logo_gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)

        # 模板匹配
        res = cv.matchTemplate(img_gray, logo_gray, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        return {
            "similarity": float(max_val),  # 相似度(0-1)
            "position": {
                "x": int(max_loc[0]),  # x坐标
                "y": int(max_loc[1]),  # y坐标
                "width": logo.shape[1],  # 宽度
                "height": logo.shape[0],  # 高度
            },
        }

    def compare_text_content(self, data1: dict, data2: dict) -> tuple[dict, list]:
        """详细比较文本内容

        参数:
            data1: 第一个文本数据列表
            data2: 第二个文本数据列表
        返回:
            包含差异和错误信息的字典
        """
        diff = self.analyze(data1, data2)
        diff_list = diff["raw_diff"]

        # 初始化错误分类
        errors = {
            "mis_chars": [],  # 错误字符
        }

        # 统计错误
        for i, d in enumerate(diff_list):
            if d.startswith("- "):
                errors["mis_chars"].append(
                    {
                        "char": d[2:],
                        "position": i,
                    }  # 错误的字符  # 位置
                )

        return {
            "raw_diff": diff["raw_diff"],  # 差异列表
            "html_diff": diff["html_diff"],  # HTML格式差异
            "errors": errors,  # 错误分类
            "error_count": len(errors["mis_chars"]),  # 错误数量
        }, diff["diff_boxes"]

    def generate_diff_boxes(self, diff: list, data1: dict) -> list:
        """为文本差异生成InfoBox对象

        参数:
            diff: 差异列表
            data1: 第一个文本数据列表
        返回:
            InfoBox对象列表
        """
        diff_boxes = []
        for s in range(len(diff)):
            if diff[s].startswith("- "):
                i = data1["text"].index(diff[s][2:])
                diff_boxes.append(
                    InfoBox(
                        x=data1["left"][i],
                        y=data1["top"][i],
                        w=data1["width"][i],
                        h=data1["height"][i],
                        color="red",  # 颜色
                    )
                )
                if s < len(diff) - 1 and diff[s + 1].startswith("+ "):
                    diff_boxes[-1].text = diff[s + 1][2:]
                    s += 1
        return diff_boxes

    def analyze(
        self,
        data1: dict,
        data2: dict,
    ) -> Dict:
        """综合图像和文本比较

        参数:
            data1: 第一个文本列表
            data2: 第二个文本列表
        返回:
            包含比较结果的字典
        """
        results = {}

        # 文本比较
        html_diff, raw_diff = self.compare_text(data1, data2)
        results["html_diff"] = html_diff
        results["raw_diff"] = raw_diff
        results["diff_boxes"] = self.generate_diff_boxes(raw_diff, data1)

        return results
