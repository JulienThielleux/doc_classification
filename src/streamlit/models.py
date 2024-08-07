import joblib
import pickle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

def CNN_model():
    cnn_model = models.efficientnet_b1()
    cnn_model.classifier[1] = nn.Linear(cnn_model.classifier[1].in_features, 16)
    cnn_model.load_state_dict(torch.load('models/cnn_model_b1_V2.pth'))

    return cnn_model


def CNN_model_proba(cnn_model, image):
    cnn_model.eval()
    with torch.no_grad():
        cnn_output = cnn_model(image)
        probabilities = F.softmax(cnn_output, dim=1)

    return probabilities


def SVC_model(text):
    with open('./models/svc_model.pkl', 'rb') as file:
        svc_model = pickle.load(file)

    return svc_model.predict([text])

def RF_model(text):
    with open('./models/rf_model.pkl', 'rb') as file:
        rf_model = joblib.load(file)

    return rf_model.predict([text])

def LR_model(text):
    with open('./models/lr_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    return lr_model.predict([text])

def LR_model_proba(text):
    with open('./models/lr_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    return lr_model.predict_proba([text])

def voting(nlp_proba, cnn_proba):
    weights = [0.46,0.47,0.68,0.7,0.47,0.46,0.65,0.26,0.81,0.55,0.68,0.19,0.14,0.11,0.13,0.34]
    inverse_weights = [1-w for w in weights]
    voting_proba = weights*cnn_proba + inverse_weights*nlp_proba
    prediction = voting_proba.argmax()

    return prediction