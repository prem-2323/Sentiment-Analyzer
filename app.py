from pathlib import Path
import pickle

from flask import Flask, jsonify, request, send_from_directory
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"

if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        "Missing model artifacts. Run Sentiment.py first to create model.h5 and tokenizer.pkl."
    )

# Load model & tokenizer
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

maxlen = 200

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get('text', '')).strip()

    if not text:
        return jsonify({"error": "Text is required."}), 400

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen)

    prediction = model.predict(pad)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"

    return jsonify({
        "sentiment": sentiment,
        "score": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)