from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import re

def resize_image(image_path, size):
    with Image.open(image_path) as img:
        img = img.resize(size)
        return img
    
def ocr_pytesseract(pic):
    ocr_text_tess = pytesseract.image_to_string(pic, config='--psm 3')
    ocr_text_tess = re.sub(r'\n', ' ', ocr_text_tess)

    return ocr_text_tess