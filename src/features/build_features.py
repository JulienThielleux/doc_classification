import cv2
from deskew import determine_skew
import numpy as np
import os
import pandas as pd
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage.transform import rotate
import spacy
import torchvision.transforms as transforms
import torch
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

from skimage import io, img_as_ubyte


def parse_xml(xml_path):
    text = ""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for page in root:
        for TextRegion in page:
            for TextLine in TextRegion:
                for Word in TextLine:
                    for TextEquiv in Word:
                        for Unicode in TextEquiv:
                            text += Unicode.text + " "    
    return text

def get_black_percent(picture):
    img_array = picture.load()
    black = 0
    for i in range(picture.size[0]):
        for j in range(picture.size[1]):
            if img_array[i,j] == 0:
                black += 1
    return (picture, black/(picture.size[0]*picture.size[1]))

def get_orientation(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        angle_moyen = np.mean(angles)
        return angle_moyen
    else:
        return None
    
def get_letter_size(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    heights = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        heights.append(h)

    average_height = np.median(heights)
    return average_height

def text_cleaning(text):
    #removing special characters
    clean_text = re.sub('[^a-zA-Z0-9]', ' ', text)
    #uncapitalizing all words
    clean_text = clean_text.lower()
    #removing one letter words
    clean_text = re.sub(r'\b\w\b', ' ', clean_text)
    #removing numbers and words containing numbers
    clean_text = re.sub('\w*\d\w*', ' ', clean_text)
    #removing stop words
    nlp = spacy.load('en_core_web_sm')
    clean_text = ' '.join([word.text for word in nlp(clean_text) if not word.is_stop])
    #removing all 2 letters words
    clean_text = re.sub(r'\b\w{1,2}\b', ' ', clean_text)
    #removing all the words that don't exist in the english dictionary
    clean_text = ' '.join([word for word in clean_text.split() if word in nlp.vocab])
    #lemmatizing
    clean_text = ' '.join([word.lemma_ for word in nlp(clean_text)])
    #shortening multiple spaces
    clean_text = re.sub(' +', ' ', clean_text)

    return clean_text

def clean_text(text):
    clean_text_list = []
    for text in tqdm(text):
        clean_text_list.append(text_cleaning(text))
    
    #building a vocabulary
    vocab = []
    for text in clean_text_list:
        vocab += text.split()
    vocab_freq = pd.Series(vocab).value_counts()
    print(f"Nombre de mots dans le vocabulaire avant suppression des moins frequents: {len(vocab_freq)}")

    #removing from the vocabulary the words that appear less than n times
    n = 2
    print(f"Suppression des mots qui apparaissent moins de {n} fois.")
    vocab_freq = vocab_freq[vocab_freq > n]
    vocab = vocab_freq.index.tolist()

    #removing all the words that are not in the vocabulary
    clean_text_list = [' '.join([word for word in text.split() if word in vocab]) for text in clean_text_list]
    print(f"Nombre de mots dans le vocabulaire apres suppression des moins frequents: {len(vocab)}")

    return clean_text_list

def similarity_score(text1, text2):
    nlp = spacy.load('en_core_web_sm')
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def jaccard_index(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if len(union) == 0:
        jaccard_index = 0
    else:
        jaccard_index = len(intersection)/len(union)

    return jaccard_index

def tfidf_similarity(corpus, text1, text2):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    tfidf_matrix = vectorizer.transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def resize_image(image):
    return cv2.resize(image, (1000, 1000))

def image_resize(img_array):
    image = Image.fromarray(img_array)
    image = image.resize((1000, 1000))
    image = np.array(image)
    return image

def correct_rotation(image, confidence_threshold: int = 10):
    pil_image = Image.fromarray(image)
    osd = pytesseract.image_to_osd(pil_image, output_type=pytesseract.Output.DICT)
    rotate_angle = osd['rotate']
    orientation_conf = osd['orientation_conf']
    
    if orientation_conf < confidence_threshold:
        rotate_angle = 0
    
    corrected_pil_image = pil_image.rotate(-rotate_angle, expand=True)
    return np.array(corrected_pil_image), osd

def deskew(image, resize: bool = True):
    angle = determine_skew(image)
    rotated = rotate(image, angle, resize=resize)
    return img_as_ubyte(np.clip(rotated, 0, 1))

def has_enough_text(image, word_threshold: int = 10):
    text = pytesseract.image_to_string(Image.fromarray(image))
    return len(text.strip().split()) > word_threshold

def has_too_many_black_pixels(image, black_pixel_threshold: float = 0.5, threshold_level: int = 10):
    pixels = np.asarray(image)
    black_pixel_ratio = np.sum(pixels < threshold_level) / pixels.size
    return black_pixel_ratio > black_pixel_threshold

def cnn_process_image(image):
    try:
        if has_enough_text(image) and not has_too_many_black_pixels(image):
            try:
                image, osd = correct_rotation(image)
            except Exception:
                pass
            image = deskew(image)
            image = resize_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image = Image.fromarray(image)
            image = transform(image)
            image = image.unsqueeze(0)
            return True, None, image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image = Image.fromarray(image)
            image = transform(image)
            image = image.unsqueeze(0)
            return False, "Filtered", image
    except Exception as e:
        return False, str(e), None
    
def ocr_pytesseract(pic):
    pic = Image.fromarray(pic)
    ocr_text_tess = pytesseract.image_to_string(pic, config='--psm 3')
    ocr_text_tess = re.sub(r'\n', ' ', ocr_text_tess)

    return ocr_text_tess

def pytesseract_ocr(pic_series):
    ocr_text_tess_list = []
    for img_array in tqdm(pic_series):
        #transform the numpy array to a picture
        pic = Image.fromarray(img_array)
        ocr_text_tess = pytesseract.image_to_string(pic, config='--psm 3')
        ocr_text_tess = re.sub(r'\n', ' ', ocr_text_tess)
        ocr_text_tess_list.append(ocr_text_tess)

    return ocr_text_tess_list