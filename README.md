# Used Mobile Resale Price Predictor

This project predicts the resale value of a used mobile phone using a machine learning model trained on synthetic market data. It is designed as a small data science and web app project that demonstrates how feature-based pricing can be estimated from real-world attributes such as brand, RAM, storage, battery health, device age, and condition.

Created by: Archit Bishnoi

## Project overview

The application allows a user to choose a phone from a searchable catalog of 120 popular models across 25 brands, or enter a custom brand/model, and provide:
- Country and display currency
- An estimated catalog reference price or a custom price
- RAM size
- Storage size
- 5G availability
- Months used
- Screen condition
- Body condition
- Battery health
- Warranty status
- Box/bill status

Based on these values, the model estimates the expected resale price in the selected currency.

The bundled model catalog is a curated starter list, not an exhaustive list of every regional SKU. Reference prices are approximate INR list-price estimates converted with fixed reference rates; they are not live retailer prices or foreign-exchange quotes. Unlisted devices can use a manually entered price.

The custom-price field starts with the selected model's estimated local reference price when available; for unlisted models it starts with a general INR-based amount converted to the selected currency. You can replace either value with your own.

## How it works

1. A synthetic dataset is generated in `venv/generate_dataset.py`.
2. The generated data includes features like brand, RAM, storage, battery health, and condition.
3. A `RandomForestRegressor` from scikit-learn is trained on this data.
4. The trained model is used inside a Streamlit web app in `app.py`.
5. The Vercel FastAPI app in `api/index.py` serves a responsive phone valuation page and prediction API.
6. The user selects a country, brand, model and price source, then receives a predicted resale price.

## Tools and technologies used

- Python
- Pandas
- NumPy
- scikit-learn
- Joblib
- Streamlit
- Matplotlib
- VS Code
- Git and GitHub

## Project structure

- `app.py` — Streamlit web app for price prediction
- `api/index.py` — FastAPI web app and prediction endpoints for Vercel
- `device_catalog.json` — phone brand/model reference catalog and country currency settings
- `venv/generate_dataset.py` — synthetic data generation script
- `used_phones.csv` — generated phone dataset
- `used_accessories.csv` — generated accessory dataset
- `README.md` — project documentation

## Run locally

From the project folder:

```bash
python venv/generate_dataset.py
streamlit run app.py
```

## Deploy to Vercel

The Vercel deployment runs the FastAPI app in `api/index.py` (the local
Streamlit app in `app.py` is separate). Import this GitHub repository into
Vercel and set the project root to the repository root. `vercel.json`
explicitly configures the Python runtime and routes requests to the FastAPI
app; `requirements.txt` lists its dependencies. After pushing deployment
changes to the connected GitHub branch, wait for Vercel to finish a new
deployment before opening the production domain.

The prediction endpoint is available at `/api/predict` and accepts a JSON POST
request. The `/api/catalog` endpoint returns available countries, brands and
models. The `/` route serves the predictor page.

You can also generate a custom dataset size:

```bash
python venv/generate_dataset.py --phones 2000 --accessories 1200 --output-dir .
```

## Notes

This project uses synthetic data for educational and demo purposes. In a real-world deployment, you would replace it with actual resale records from a marketplace or a curated business dataset.
