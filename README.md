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

This project can publish the frontend as a static site on Netlify and the backend on Render.

1. Deploy the backend to Render using `render.yaml`.
2. Replace `https://YOUR-RENDER-SERVICE.onrender.com` in `netlify.toml` with your real Render service URL.
3. Deploy the repository root to Netlify with the included `netlify.toml`.
4. Netlify will proxy `/api/predict` to the Render backend, so the frontend and backend stay connected.

## Notes

- Latest recorded test accuracy: 92.20%.
- The model saves as `model.h5`, which is a legacy HDF5 format but works with this project.
- The TensorFlow GPU warning on native Windows is expected with modern TensorFlow versions and does not block CPU training.