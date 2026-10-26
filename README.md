# Used Mobile Resale Price Predictor

This project predicts the resale value of a used mobile phone using a machine learning model trained on synthetic market data. It is designed as a small data science and web app project that demonstrates how feature-based pricing can be estimated from real-world attributes such as brand, RAM, storage, battery health, device age, and condition.

Created by: Archit Bishnoi

## Project overview

The application allows a user to enter mobile details like:
- Brand
- Original purchase price
- RAM size
- Storage size
- 5G availability
- Months used
- Screen condition
- Body condition
- Battery health
- Warranty status
- Box/bill status

Based on these values, the model estimates the expected resale price in INR.

## How it works

1. A synthetic dataset is generated in `venv/generate_dataset.py`.
2. The generated data includes features like brand, RAM, storage, battery health, and condition.
3. A `RandomForestRegressor` from scikit-learn is trained on this data.
4. The trained model is used inside a Streamlit web app in `app.py`.
5. The user enters values in the UI and receives a predicted resale price instantly.

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

The Vercel deployment uses the FastAPI app in `api/index.py` (the local
Streamlit app is separate). Import this GitHub repository into Vercel and keep
the project root set to the repository root. Vercel uses `vercel.json` and
`requirements.txt` to configure the Python function. After pushing deployment
changes to the connected GitHub branch, wait for Vercel to finish a new
deployment before opening the production domain.

The prediction endpoint is available at `/api/predict` and accepts a JSON POST
request. The `/` route serves the predictor page.

You can also generate a custom dataset size:

```bash
python venv/generate_dataset.py --phones 2000 --accessories 1200 --output-dir .
```

## Notes

This project uses synthetic data for educational and demo purposes. In a real-world deployment, you would replace it with actual resale records from a marketplace or a curated business dataset.
