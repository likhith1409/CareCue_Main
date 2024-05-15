import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


metadata = pd.read_csv('medicine_dataset/metadata.csv')

def load_images_from_metadata(metadata):
    images = []
    labels = []
    for index, row in metadata.iterrows():
        img_path = os.path.join('medicine_dataset', row['filepath'])
        if not os.path.exists(img_path):
            print("File not found:", img_path)
            continue
        img = cv2.imread(img_path)
        if img is None:
            print("Unable to read file:", img_path)
            continue
        img = cv2.resize(img, (64, 64))  
        img = img.flatten()  
        images.append(img)
        labels.append(row['label'])  
    return images, labels


images, labels = load_images_from_metadata(metadata)


train_images, validation_images, train_labels, validation_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


train_images = np.array(train_images)
train_labels = np.array(train_labels)
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)


model = LogisticRegression()
model.fit(train_images, train_labels)


validation_predictions = model.predict(validation_images)
print('Validation Accuracy:', accuracy_score(validation_labels, validation_predictions))


joblib.dump(model, 'logistic_regression_model.pkl')
