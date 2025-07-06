import re
import cv2 as cv

import pytesseract
from pytesseract import Output


def deal_text(img, meth: str) -> dict:
    if img is None:
        raise ValueError("No Image")
    src = pytesseract.image_to_data(
        img,
        lang="eng",
        config="--psm 1",
        output_type=Output.DICT,
    )
    data = {}
    for k in src.keys():
        data[k] = []
    # 过滤-1置信度的文字
    # 过滤非白名单的文字
    whitelist = ["A-Z0-9", "():\\./\-"]
    for i, (cf, text) in enumerate(zip(src["conf"], src["text"])):
        if cf == -1:
            continue
        text = re.sub(r"[^" + "".join(whitelist) + "]", "", text)
        if len(text) == 0:
            continue
        src["text"][i] = text
        for k, v in src.items():
            data[k].append(v[i])

    total_conf = 0
    for i, cf in enumerate(data["conf"]):
        total_conf += cf
    average_conf = total_conf / len(data["conf"]) if len(data["conf"]) > 0 else 0
    data["total_conf"] = total_conf
    data["average_conf"] = average_conf
    data["text_num"] = len(data["text"])
    data["meth"] = meth
    return data


def max_conf_text(img, name) -> dict:
    """处理带有文字的图片，返回最大置信度的OCR识别结果

    Args:
        img (np.ndarray): 带有文字的图片
        name (str): 图片名称

    Returns:
        Dict: 最大置信度的OCR识别结果
    """
    _data = []
    _data.append(deal_text(img, "non"))

    adpt = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 7
    )
    dil = cv.dilate(adpt, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    ero = cv.erode(dil, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    _data.append(deal_text(adpt, "adpt"))
    _data.append(deal_text(dil, "adpt_dil"))
    _data.append(deal_text(ero, "adpt_dil_ero"))

    otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    dil = cv.dilate(otsu, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    ero = cv.erode(dil, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    _data.append(deal_text(otsu, "otsu"))
    _data.append(deal_text(dil, "otsu_dil"))
    _data.append(deal_text(ero, "otsu_dil_ero"))

    # cv.imwrite(f"{name}_ocr_pre_adpt.jpg", adpt)
    # cv.imwrite(f"{name}_ocr_pre_otsu.jpg", otsu)

    ret = _data[0]
    for data in _data:
        tmp = ret
        if data["text_num"] > ret["text_num"]:
            tmp = data
        if data["average_conf"] - ret["average_conf"] > 10:
            tmp = data
        if ret["average_conf"] - data["average_conf"] > 10:
            tmp = ret
        ret = tmp
    return ret
