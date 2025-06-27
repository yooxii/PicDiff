from PIL import Image
import pytesseract
 
text = pytesseract.image_to_string(Image.open('afterEC.png'))
print(text)