import cv2 as cv
import os
import json
import pytesseract
from pytesseract import Output

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
        poi1=(0, 0),
        poi2=(0, 0),
        color="red",
        text="",
        thickness=1,
        line_type=cv.LINE_AA,
        font_size=16,
        font_type="Arial",
        font_weight="normal",
    ):
        self.poi1 = poi1
        self.poi2 = poi2
        self.color = color
        self.text = text
        self.thickness = thickness
        self.line_type = line_type
        self.font_size = font_size
        self.font_type = font_type
        self.font_weight = font_weight


class Img:
    def __init__(self, img_path):
        self.path = img_path
        self.img = cv.imread(img_path)
        if self.img is None:
            raise ValueError("Image not found")
        self.name = os.path.basename(img_path)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.data = None

        self.max_conf_text()

    def reread(self):
        self.img = cv.imread(self.path)
        return self.img

    def resize(self, size):
        self.img = cv.resize(self.img, None, fx=size, fy=size)
        return self.img

    def img2gray(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        return self.img

    def gray2bin_OTSU(self):
        self.img = cv.threshold(self.img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        return self.img

    def gray2bin_adap(self):
        self.img = cv.adaptiveThreshold(
            self.img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 5
        )
        return self.img

    def deal_logo(self):
        return self.img

    def deal_text(self) -> dict:
        src = pytesseract.image_to_data(
            self.img,
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
        return data

    def max_conf_text(self):
        _data = []

        self.reread()
        self.img2gray()
        _data.append(self.deal_text())

        self.reread()
        self.img2gray()
        self.gray2bin_adap()
        _data.append(self.deal_text())

        self.reread()
        self.img2gray()
        self.gray2bin_OTSU()
        _data.append(self.deal_text())

        for data in _data:
            if self.data is None:
                self.data = data
                continue
            if data["text_num"] > self.data["text_num"]:
                self.data = data
            elif data["text_num"] == self.data["text_num"]:
                if data["total_conf"] > self.data["total_conf"]:
                    self.data = data

    def draw_boxes(self, boxes: list[InfoBox]):
        for box in boxes:
            cv.rectangle(
                self.img,
                box.poi1,
                box.poi2,
                COLOR[box.color],
                box.thickness,
                box.line_type,
            )
        return self.img


def list_path_images(path: str):
    imgs = []
    for file in os.listdir(path):
        if file.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")):
            imgs.append(os.path.join(path, file))
    return imgs


def deal_std(img: Img):
    return img


def deal_uut(img: Img):
    return img


def compare_images(standard: Img, uut: Img):
    standard = deal_std(standard)
    uut = deal_uut(uut)


def main():
    # uuts = list_path_images("Pics")
    # stds = list_path_images("Standards")
    # print(uuts)
    # print(stds)
    std = Img("Standards/AP23NA.png")
    uut = Img("Pics/UUT-2.jpg")

    with open("data.json", "w") as f:
        json.dump(std.data, f)
        f.write("\n")
        json.dump(uut.data, f)


if __name__ == "__main__":
    main()
