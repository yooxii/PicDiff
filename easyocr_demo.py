import easyocr

# 创建OCR对象
reader = easyocr.Reader(["en", "ja"])

# 识别文字
result = reader.readtext("Pics/UUT.png")

# 处理识别结果
for text, bbox, confidence in result:
    print(f"Text: {text}, Bbox: {bbox}, Confidence: {confidence}")
