import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils.preprocessing import load_and_clean_data
from features.feature_loader import get_handcrafted_features
from bert_model.bert_features import get_bert_embeddings


# load dataset
data = load_and_clean_data("dataset/raw/legitphish.csv")
data = data.sample(20000, random_state=42)

# labels
y = data["ClassLabel"]

# handcrafted features
handcrafted = get_handcrafted_features(data)

# BERT embeddings
urls = data["URL"]
bert_embeddings = get_bert_embeddings(urls)

# convert BERT tensor to numpy
bert_embeddings = bert_embeddings.numpy()

# combine features
X = np.concatenate((bert_embeddings, handcrafted.values), axis=1)

print("Final feature shape:", X.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train classifier
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


import joblib

joblib.dump(model, "saved_models/phishing_model.pkl")

print("Model saved successfully")
