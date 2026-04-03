# Sentiment Analyzer

Twitter sentiment analysis using a SimpleRNN model trained on the provided `twitter_training.csv` dataset and served through a Flask web app.

## Files

- `Sentiment.py` trains the model and saves `model.h5` and `tokenizer.pkl`.
- `app.py` runs the Flask backend and exposes the `/predict` endpoint.
- `index.html` provides the frontend UI.
- `twitter_training.csv` is the training dataset.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Train the model

Run the training script once to create the saved model artifacts:

```bash
python Sentiment.py
```

## Start the web app

After training completes, launch the Flask app:

```bash
python app.py
```

Then open the local server in your browser at `http://127.0.0.1:5000`.

## Deploy the frontend on Netlify

This project can publish the frontend as a static site on Netlify.

1. Set the backend URL in `index.html` by filling the `api-base-url` meta tag with your Flask app URL.
2. Deploy the repository root to Netlify with the included `netlify.toml`.
3. If you only want the static UI, Netlify can host `index.html` directly, but the prediction button still needs a live Flask backend.

## Notes

- Latest recorded test accuracy: 92.20%.
- The model saves as `model.h5`, which is a legacy HDF5 format but works with this project.
- The TensorFlow GPU warning on native Windows is expected with modern TensorFlow versions and does not block CPU training.