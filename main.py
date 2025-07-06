import os
import cv2 as cv
import numpy as np
import json
from picdiff_modules.iimg import Img
from picdiff_modules.html_report import HTMLReportGenerator


def main():
    try:
        std_img = Img("Pics/AP23NA-01.jpg")
        uut_img = Img("Pics/UUT-2.jpg")

        std_img.resize_to_standard()
        std_img.preprocess_image()

        uut_img.resize_to_standard()
        uut_img.preprocess_image()

        results, diff_boxes = uut_img.compare_with(std_img)

        print("\n=== Logo Comparison ===")
        if results["logo_comparison"]:
            print(f"Similarity: {results['logo_comparison']['similarity']:.2%}")
            print(
                f"Position: x={results['logo_comparison']['position']['x']}, y={results['logo_comparison']['position']['y']}"
            )
        else:
            print("No logo comparison performed (missing logo files)")

        print("\n=== Text Comparison ===")
        print(f"Total errors found: {results['text_comparison']['error_count']}")
        print("Error details:")
        for err_type, errors in results["text_comparison"]["errors"].items():
            if errors:
                print(f"- {err_type}: {len(errors)}")

        with open("comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        if uut_img.text_img is not None:
            cv.imwrite("text_boxes.jpg", uut_img.img)

        print("Image comparison completed successfully")
        print(f"- Visual differences saved to diff_result.jpg")
        print(f"- Text differences saved to diff_data.json")

        # 生成HTML报告
        report_gen = HTMLReportGenerator()

        # 直接使用diff_boxes和raw_diff生成报告
        report_path = report_gen.generate_html(
            uut_img.output_img,
            results["text_comparison"],
            diff_boxes,
            "image_comparison_report",
        )

        print(f"- HTML report generated at {report_path}")

    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()
