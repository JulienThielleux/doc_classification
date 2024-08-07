import cv2
from PIL import Image
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
import re
import spacy
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def resize_image(image_path, size):
    with Image.open(image_path) as img:
        img = img.resize(size)
        return img
    
def ocr_pytesseract(pic):
    ocr_text_tess = pytesseract.image_to_string(pic, config='--psm 3')
    ocr_text_tess = re.sub(r'\n', ' ', ocr_text_tess)

    return ocr_text_tess

def clean_text(text):
    clean_text = re.sub('[^a-zA-Z0-9]', ' ', text)
    clean_text = clean_text.lower()
    clean_text = re.sub(r'\b\w\b', ' ', clean_text)
    clean_text = re.sub('\w*\d\w*', ' ', clean_text)
    nlp = spacy.load('en_core_web_sm')
    clean_text = ' '.join([word.text for word in nlp(clean_text) if not word.is_stop])
    clean_text = re.sub(r'\b\w{1,2}\b', ' ', clean_text)
    clean_text = ' '.join([word for word in clean_text.split() if word in nlp.vocab])
    clean_text = ' '.join([word.lemma_ for word in nlp(clean_text)])
    clean_text = re.sub(' +', ' ', clean_text)
    
    return clean_text


def to_class_name(class_num):

    class_name_dict = {
        0: "lettre",
        1: "formulaire",
        2: "email",
        3: "manuscrit",
        4: "publicité",
        5: "rapport scientifique",
        6: "publication scientifique",
        7: "spécification",
        8: "dossier",
        9: "article de presse",
        10: "budget",
        11: "facture",
        12: "présentation",
        13: "questionnaire",
        14: "CV",
        15: "note de service"
        }

    return class_name_dict[class_num]

def grad_cam(model, image_tensor, target_layer_name):
    gradients = []

    def save_gradient(grad):
        gradients.append(grad)

    target_layer = dict([*model.named_modules()])[target_layer_name]
    activations = []

    def save_activation(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = target_layer.register_forward_hook(save_activation)

    # Move image tensor to the same device as the model
    image_tensor = image_tensor.to(next(model.parameters()).device)

    # Forward pass
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[:, pred_class].backward()

    handle.remove()  # Remove the forward hook

    gradients = gradients[0].cpu().data.numpy()
    activations = activations[0].cpu().data.numpy()

    return gradients, activations, pred_class

def generate_heatmap(gradients, activations):
    weights = np.mean(gradients, axis=(2, 3))
    cam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

def superimpose_heatmap(image_tensor, heatmap):
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = np.uint8(255 * image)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_PLASMA)
    superimposed_img = heatmap * 0.05 + image
    return superimposed_img

# Extract layers
def get_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            layers.append(name)
    return layers

# Function to preprocess the image
def preprocess_image(image_path):

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    return image