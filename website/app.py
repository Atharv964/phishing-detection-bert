from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import torch

from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Load trained model
model = joblib.load("../saved_models/phishing_model.pkl")

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def get_bert_features(url):

    inputs = tokenizer(
        url,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=32
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    embedding = outputs.last_hidden_state[:,0,:].numpy()

    return embedding

def extract_url_features(url):

    url_length = len(url)
    dot_count = url.count(".")
    has_https = 1 if "https" in url else 0
    has_ip = 1 if "://" in url and any(char.isdigit() for char in url) else 0

    features = [
        url_length,
        has_ip,
        dot_count,
        has_https
    ]

    while len(features) < 16:
        features.append(0)

    return np.array(features).reshape(1,-1)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    url = request.form["url"]

    bert_features = get_bert_features(url)
    url_features = extract_url_features(url)

    features = np.concatenate((bert_features, url_features), axis=1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Legitimate Website ✅"
    else:
        result = "Phishing Website ⚠️"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)