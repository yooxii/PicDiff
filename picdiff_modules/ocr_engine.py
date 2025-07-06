import re
import pytesseract
from pytesseract import Output
import cv2 as cv
import numpy as np
from typing import Dict, List


class OCREngine:
    def __init__(self):
        self.whitelist = ["A-Z0-9", "():\\./\\-"]
        self.methods = ["non", "adpt", "otsu"]

    def preprocess_image(self, img: np.ndarray, method: str) -> np.ndarray:
        """根据指定方法预处理图像用于OCR识别"""
        # 确保输入是3通道BGR图像
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if method == "non":
            return gray
        elif method == "adpt":
            return cv.adaptiveThreshold(
                gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 7
            )
        elif method == "otsu":
            _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            return thresh
        return gray

    def clean_text(self, text: str) -> str:
        """使用白名单清理OCR识别结果"""
        return re.sub(r"[^" + "".join(self.whitelist) + "]", "", text)

    def _detect_text_regions(self, img: np.ndarray) -> List[tuple]:
        """检测图像中的文字区域，特别针对大的文本块"""
        # 确保输入是3通道BGR图像
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        # 过滤小噪点
        img = cv.medianBlur(img, 7)

        # 转换为灰度图
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 应用自适应阈值
        thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
        )

        # 形态学操作增强文本区域
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 13))
        dilated = cv.dilate(thresh, kernel, iterations=3)

        # 查找轮廓
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # 过滤并合并相邻的文字区域
        text_regions = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            # 只保留较大的区域(宽度大于图像宽度的1/3)
            if w > img.shape[1] // 3 and h > 20:
                # 检查是否可以与已有区域合并
                merged = False
                for i, (rx, ry, rw, rh) in enumerate(text_regions):
                    # 如果区域相邻或重叠
                    if abs(x - rx) < 50 and abs(y - ry) < 50:
                        # 合并区域
                        new_x = min(x, rx)
                        new_y = min(y, ry)
                        new_w = max(x + w, rx + rw) - new_x
                        new_h = max(y + h, ry + rh) - new_y
                        text_regions[i] = (new_x, new_y, new_w, new_h)
                        merged = True
                        break
                if not merged:
                    text_regions.append((x, y, w, h))

        # 如果没有检测到大区域，则使用原始方法
        if not text_regions:
            contours, _ = cv.findContours(
                thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                if w > 50 and h > 10:  # 放宽条件
                    text_regions.append((x, y, w, h))

        return text_regions

    def process_image(self, img: np.ndarray, method: str) -> Dict:
        """使用OCR处理图像并返回结构化数据"""
        # 首先检测文字区域
        text_regions = self._detect_text_regions(img)

        # 初始化结果字典
        results = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
        }

        # 分别处理每个文字区域
        for x, y, w, h in text_regions:
            # 裁剪文字区域
            region = img[y : y + h, x : x + w]

            # 预处理并OCR识别该区域
            processed_img = self.preprocess_image(region, method)
            raw_data = pytesseract.image_to_data(
                processed_img,
                lang="eng+chi_sim",
                config="--psm 6",
                output_type=Output.DICT,
            )

            # 处理OCR结果
            total_conf = 0
            valid_count = 0

            for i, (conf, text) in enumerate(zip(raw_data["conf"], raw_data["text"])):
                if conf == -1 or not text.strip():
                    continue

                cleaned_text = self.clean_text(text)
                if not cleaned_text:
                    continue

                # 调整坐标到原图位置
                results["text"].append(cleaned_text)
                results["left"].append(x + raw_data["left"][i])
                results["top"].append(y + raw_data["top"][i])
                results["width"].append(raw_data["width"][i])
                results["height"].append(raw_data["height"][i])
                results["conf"].append(conf)

                total_conf += conf
                valid_count += 1

            # 计算置信度指标
            results["total_conf"] = results.get("total_conf", 0) + total_conf
            results["average_conf"] = total_conf / valid_count if valid_count > 0 else 0
            results["text_num"] = results.get("text_num", 0) + valid_count
            results["meth"] = method

        return results

    def get_best_ocr_result(self, img: np.ndarray) -> Dict:
        """尝试多种方法获取最佳OCR结果"""
        results = []
        for method in self.methods:
            results.append(self.process_image(img, method))

        best_result = results[0]
        for result in results[1:]:
            if result["average_conf"] > best_result["average_conf"] + 10:
                best_result = result

        return best_result
