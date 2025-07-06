import cv2 as cv
import numpy as np

COLOR = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
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


def draw_boxes(img, boxes: list[InfoBox]):
    if img is None:
        raise ValueError("No Image")
    for box in boxes:
        color = COLOR[box.color] if isinstance(box.color, str) else box.color
        cv.rectangle(
            img=img,
            pt1=box.p1,
            pt2=box.p2,
            color=color,
            thickness=box.thickness,
            lineType=box.line_type,
        )
        h = img.shape[0]
        cv.putText(
            img,
            box.text,
            (box.x + box.w // 2 - 30, box.y + box.h),
            cv.FONT_HERSHEY_SIMPLEX,
            h / 700,
            COLOR["black"],
            2,
            cv.LINE_AA,
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
