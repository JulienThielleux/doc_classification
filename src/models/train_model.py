from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


def train_rf_model(data, labels):
    text_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
    ])

    text_clf.fit(data, labels)

    return text_clf

def train_lr_model(data, labels):
    text_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', LogisticRegression())
    ])

    text_clf.fit(data, labels)

    return text_clf

def train_svc_model(data, labels):
    text_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', LinearSVC())
    ])

    text_clf.fit(data, labels)

    return text_clf


def cnn_selection(type = 'b1', NUM_CLASSES = 16):
    """
    Forward passes the image tensor batch through the model 
    `n_runs` number if times. Prints the average milliseconds taken
    and returns a list containing all the forward pass times.
    """
    if type == 'b1' :
        cnn_model = models.efficientnet_b1(pretrained=True)
    elif type == 'b0' :
        cnn_model = models.efficientnet_b0(pretrained=True)
    cnn_model.classifier[1] = nn.Linear(cnn_model.classifier[1].in_features, NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)
    return cnn_model


def train_cnn_model(cnn_model, input_batch, n_epochs = 30):

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(n_epochs):
        cnn_model.train()
        running_loss = 0.0
        for images, labels in input_batch:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return cnn_model

