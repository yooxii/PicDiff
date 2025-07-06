import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path
from .image_utils import InfoBox


class HTMLReportGenerator:
    """生成包含标注图像和文本对比结果的HTML报告"""

    ERROR_COLORS = {
        "text_mismatch": (0, 0, 255),  # 红色 - 文本不匹配
        "missing_text": (255, 0, 0),  # 蓝色 - 缺失文本
        "extra_text": (0, 255, 0),  # 绿色 - 多余文本
        "position_error": (255, 255, 0),  # 黄色 - 位置错误
    }

    def __init__(self, output_dir: str = "reports"):
        """初始化报告生成器

        参数:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def annotate_image(
        self, image: np.ndarray, diff_boxes: List[InfoBox]
    ) -> np.ndarray:
        """在图像上标注错误区域

        参数:
            image: 原始图像
            diff_boxes: 差异框信息列表
        返回:
            标注后的图像
        """
        annotated = image.copy()

        # 标注错误区域
        for box in diff_boxes:
            x, y, w, h = box.x, box.y, box.w, box.h
            color = (0, 0, 255)  # 红色标注错误区域
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # 添加错误文本
            cv2.putText(
                annotated,
                box.text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # 添加图例
        legend_x = image.shape[1] - 200
        legend_y = 40
        cv2.putText(
            annotated,
            "Error Regions",
            (legend_x, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            1,
        )
        cv2.rectangle(
            annotated,
            (legend_x, legend_y + 80),
            (legend_x + 80, legend_y + 160),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            annotated,
            "Text Errors",
            (legend_x + 100, legend_y + 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            1,
        )

        return annotated

    def generate_html(
        self,
        image: np.ndarray,
        diff_data: Dict,
        diff_boxes: List,
        report_name: str = "report",
    ) -> str:
        """生成HTML报告

        参数:
            image_path: 标注图像路径
            diff_data: 差异分析结果
            report_name: 报告名称
        返回:
            生成的HTML文件路径
        """
        # 读取并标注图像
        annotated = self.annotate_image(image, diff_boxes)

        # 保存标注图像
        annotated_path = str(self.output_dir / f"{report_name}_annotated.png")
        cv2.imwrite(annotated_path, annotated)

        # 生成HTML
        html_path = str(self.output_dir / f"{report_name}.html")
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
        img {{ max-width: 100%; height: auto; }}
        .error {{ margin: 5px 0; padding: 5px; }}
    </style>
</head>
<body>
    <h1>PicDiff Report - {report_name}</h1>
    <div class="container">
        <div class="text-comparison">
            <h2>Text Comparison</h2>
            <pre>{diff_data.get("html_diff", "")}</pre>
        </div>
        <div class="image-container">
            <h2>Annotated Image</h2>
            <img src="{report_name}_annotated.png" alt="Annotated Image">
        </div>
    </div>
</body>
</html>
            """
            )

        return html_path
