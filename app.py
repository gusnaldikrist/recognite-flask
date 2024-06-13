from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model
model_path = 'model/model.h5'  # Update this path if necessary
model = load_model(model_path)

# Load and preprocess the dataset for tokenizer and label encoder
data = pd.read_csv('dataset/dataset.csv')
data['cleaned_text'] = data['cleaned_text'].astype(str)
data['cleaned_text'].fillna('', inplace=True)

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(data['cleaned_text'].values)

# Determine the maxlen used during training
X = tokenizer.texts_to_sequences(data['cleaned_text'].values)
X = pad_sequences(X)
maxlen = X.shape[1]

# Define the prediction function
def predict_sentiment(tweet):
    tweet_seq = tokenizer.texts_to_sequences([tweet])
    tweet_pad = pad_sequences(tweet_seq, maxlen=maxlen)
    prediction = model.predict(tweet_pad)
    predicted_label = np.argmax(prediction, axis=1)
    original_label = label_encoder.inverse_transform(predicted_label)
    return original_label[0]

# Create a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.json or 'text' not in request.json:
            return jsonify({'error': 'No text provided'}), 400

        tweet = request.json['text']
        predicted_label = predict_sentiment(tweet)
        return jsonify({'predicted_label': str(predicted_label)})  # Convert to string
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
