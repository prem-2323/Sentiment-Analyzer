# RNN (SimpleRNN) - Sentiment Analysis using Twitter Dataset

from pathlib import Path
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.utils import pad_sequences
from keras.src.legacy.preprocessing.text import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"

# Load Twitter dataset (no header: id, topic, label, text)
df = pd.read_csv(
    "twitter_training.csv",
    header=None,
    names=["id", "topic", "label", "text"]
)

# Keep only binary sentiment rows and map labels to 0/1
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(str).str.lower()
df = df[df["label"].isin(["negative", "positive"])]

texts = df["text"].astype(str)
labels = df["label"].map({"negative": 0, "positive": 1}).astype(int)


def augment_text(text):
    words = text.split()
    if len(words) < 4:
        return text

    # Randomly drop a few words to create a slightly different variant.
    kept_words = [w for w in words if random.random() > 0.1]
    if len(kept_words) < 2:
        kept_words = words
    return " ".join(kept_words)


# Increase data size by adding one augmented sample per tweet.
augmented_texts = texts.tolist()
augmented_labels = labels.tolist()

for t, l in zip(texts, labels):
    augmented_texts.append(augment_text(t))
    augmented_labels.append(l)

augmented_labels = np.array(augmented_labels, dtype=np.int32)

# Tokenization
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(augmented_texts)

sequences = tokenizer.texts_to_sequences(augmented_texts)

# Padding
maxlen = 200
padded = pad_sequences(sequences, maxlen=maxlen)

# Split
split = int(0.8 * len(padded))
x_train, x_test = padded[:split], padded[split:]
y_train, y_test = augmented_labels[:split], augmented_labels[split:]

# Build SimpleRNN model
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128),
    layers.SimpleRNN(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    layers.SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", test_acc)

# Save trained artifacts for the Flask app.
model.save(MODEL_PATH)
with open(TOKENIZER_PATH, "wb") as file:
    pickle.dump(tokenizer, file)

print("Saved model to:", MODEL_PATH)
print("Saved tokenizer to:", TOKENIZER_PATH)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict custom tweet
custom_text = ["excellent movie"]

seq = tokenizer.texts_to_sequences(custom_text)
pad = pad_sequences(seq, maxlen=200)

prediction = model.predict(pad, verbose=0)[0][0]
sentiment = "Positive" if prediction > 0.5 else "Negative"

print("Tweet:", custom_text[0])
print("Predicted Sentiment:", sentiment, "(", round(prediction, 2), ")")