import cv2 as cv
import numpy as np
from typing import Optional, Tuple, List


class InfoBox:
    def __init__(
        self,
        x: int = None,
        y: int = None,
        w: int = None,
        h: int = None,
        p1: Tuple[int, int] = (0, 0),
        p2: Tuple[int, int] = (0, 0),
        color: str = "red",
        text: str = "",
        thickness: int = 1,
        line_type: int = cv.LINE_AA,
    ):
        """信息框类，用于在图像上绘制矩形框和文本

        参数:
            x: 左上角x坐标
            y: 左上角y坐标
            w: 宽度
            h: 高度
            p1: 第一个点坐标(可选)
            p2: 第二个点坐标(可选)
            color: 颜色名称或BGR元组
            text: 框内显示的文本
            thickness: 线宽
            line_type: 线型
        """
        if None in [x, y, w, h] and p1 != p2:
            # 根据两点坐标计算矩形框位置和大小
            self.x = min(p1[0], p2[0])
            self.y = min(p1[1], p2[1])
            self.w = abs(p1[0] - p2[0])
            self.h = abs(p1[1] - p2[1])
        elif p1 == p2 and None not in [x, y, w, h]:
            # 直接使用坐标和宽高
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.p1 = (x, y)
            self.p2 = (x + w, y + h)
        else:
            raise ValueError("无效的矩形框坐标")

        self.color = color  # 颜色
        self.text = text  # 文本内容
        self.thickness = thickness  # 线宽
        self.line_type = line_type  # 线型


def draw_boxes(img: np.ndarray, boxes: List[InfoBox]) -> np.ndarray:
    """在图像上绘制多个信息框

    参数:
        img: 输入图像
        boxes: InfoBox对象列表
    返回:
        绘制了矩形框的图像
    """
    if img is None:
        raise ValueError("无有效图像")

    # 颜色名称到BGR值的映射
    color_map = {
        "red": (0, 0, 255),  # 红色
        "green": (0, 255, 0),  # 绿色
        "blue": (255, 0, 0),  # 蓝色
        "yellow": (0, 255, 255),  # 黄色
        "white": (255, 255, 255),  # 白色
        "black": (0, 0, 0),  # 黑色
    }

    for box in boxes:
        # 将颜色名称转换为BGR元组(默认为红色)
        color = color_map.get(box.color.lower(), (0, 0, 255))

        # 绘制矩形框
        cv.rectangle(
            img,
            (box.x, box.y),
            (box.x + box.w, box.y + box.h),
            color,
            box.thickness,
            box.line_type,
        )
        # 如果有文本则绘制文本
        if box.text:
            cv.putText(
                img,
                box.text,
                (box.x, box.y - 5),  # 文本位置(框上方)
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,  # 字体大小
                color,
                1,  # 线宽
                cv.LINE_AA,  # 抗锯齿
            )
    return img


def list_image_files(path: str) -> List[str]:
    """列出目录中的所有图像文件

    参数:
        path: 目录路径
    返回:
        图像文件路径列表
    """
    import os

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # 支持的图像格式
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(valid_extensions)  # 检查文件扩展名
    ]
