from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / 'used_phones.csv'
MODEL_FILE = Path(tempfile.gettempdir()) / 'used_phone_price_model.joblib'


class PredictionInput(BaseModel):
    brand: str = Field(..., min_length=1)
    ram_gb: int = Field(..., ge=4, le=16)
    storage_gb: int = Field(..., ge=64, le=512)
    has_5g: int = Field(..., ge=0, le=1)
    original_price_inr: int = Field(..., gt=0)
    months_used: int = Field(..., ge=1, le=47)
    screen_condition: int = Field(..., ge=1, le=5)
    body_condition: int = Field(..., ge=1, le=5)
    battery_health_pct: int = Field(..., ge=65, le=100)
    under_warranty: int = Field(..., ge=0, le=1)
    has_box_bill: int = Field(..., ge=0, le=1)


app = FastAPI(title='Used Mobile Resale Price Predictor', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'Data file not found: {DATA_FILE}')
    return pd.read_csv(DATA_FILE)


def train_model():
    df = load_dataset()
    X = df.drop(columns=['resale_price_inr'])
    y = df['resale_price_inr']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = ['brand']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=None,
                min_samples_leaf=1,
            )),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2': float(r2_score(y_test, y_pred)),
    }
    return model, metrics


def get_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)

    model, _ = train_model()
    joblib.dump(model, MODEL_FILE)
    return model


@app.get('/', response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Used Mobile Resale Price Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                background: linear-gradient(135deg, #f6f8fb, #eef3ff);
                color: #1f2937;
            }
            .container {
                max-width: 900px;
                margin: 50px auto;
                background: white;
                border-radius: 18px;
                box-shadow: 0 20px 40px rgba(31, 41, 55, 0.08);
                padding: 40px 32px;
            }
            h1 {
                margin-top: 0;
                font-size: 2.6rem;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 18px;
            }
            .field {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            label {
                font-weight: 600;
            }
            input, select, button {
                padding: 12px 14px;
                border-radius: 10px;
                border: 1px solid #d1d5db;
                font-size: 1rem;
            }
            button {
                background: #2563eb;
                color: white;
                border: none;
                cursor: pointer;
                margin-top: 18px;
                font-weight: 700;
            }
            button:hover {
                background: #1d4ed8;
            }
            .result {
                margin-top: 28px;
                padding: 18px 20px;
                background: #ecfdf5;
                border-radius: 12px;
                border: 1px solid #bbf7d0;
                display: none;
            }
            .result strong {
                font-size: 1.5rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Used Mobile Resale Price Predictor</h1>
            <div class="grid">
                <div class="field">
                    <label for="brand">Brand</label>
                    <select id="brand">
                        <option>Apple</option>
                        <option>Samsung</option>
                        <option>OnePlus</option>
                        <option>Xiaomi</option>
                        <option>Realme</option>
                        <option>Vivo</option>
                        <option>iQOO</option>
                    </select>
                </div>
                <div class="field">
                    <label for="original_price_inr">Original Price (₹)</label>
                    <input id="original_price_inr" type="number" value="40000" min="5000" max="200000" />
                </div>
                <div class="field">
                    <label for="ram_gb">RAM (GB)</label>
                    <input id="ram_gb" type="number" value="8" min="4" max="16" step="2" />
                </div>
                <div class="field">
                    <label for="storage_gb">Storage (GB)</label>
                    <select id="storage_gb">
                        <option>64</option>
                        <option selected>128</option>
                        <option>256</option>
                        <option>512</option>
                    </select>
                </div>
                <div class="field">
                    <label for="has_5g">5G</label>
                    <select id="has_5g">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="field">
                    <label for="months_used">Months Used</label>
                    <input id="months_used" type="number" value="12" min="1" max="47" />
                </div>
                <div class="field">
                    <label for="screen_condition">Screen Condition (1-5)</label>
                    <input id="screen_condition" type="number" value="4" min="1" max="5" />
                </div>
                <div class="field">
                    <label for="body_condition">Body Condition (1-5)</label>
                    <input id="body_condition" type="number" value="4" min="1" max="5" />
                </div>
                <div class="field">
                    <label for="battery_health_pct">Battery Health (%)</label>
                    <input id="battery_health_pct" type="number" value="85" min="65" max="100" />
                </div>
                <div class="field">
                    <label for="under_warranty">Under Warranty</label>
                    <select id="under_warranty">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="field">
                    <label for="has_box_bill">Has Box/Bill</label>
                    <select id="has_box_bill">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
            </div>
            <button id="submitBtn">Predict resale price</button>
            <div class="result" id="resultBox">
                <div>Estimated resale value</div>
                <strong id="predictedValue">₹0</strong>
            </div>
        </div>

        <script>
            const submitBtn = document.getElementById('submitBtn');
            const resultBox = document.getElementById('resultBox');
            const predictedValue = document.getElementById('predictedValue');

            submitBtn.addEventListener('click', async () => {
                const payload = {
                    brand: document.getElementById('brand').value,
                    original_price_inr: Number(document.getElementById('original_price_inr').value),
                    ram_gb: Number(document.getElementById('ram_gb').value),
                    storage_gb: Number(document.getElementById('storage_gb').value),
                    has_5g: Number(document.getElementById('has_5g').value),
                    months_used: Number(document.getElementById('months_used').value),
                    screen_condition: Number(document.getElementById('screen_condition').value),
                    body_condition: Number(document.getElementById('body_condition').value),
                    battery_health_pct: Number(document.getElementById('battery_health_pct').value),
                    under_warranty: Number(document.getElementById('under_warranty').value),
                    has_box_bill: Number(document.getElementById('has_box_bill').value),
                };

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                predictedValue.textContent = `₹${new Intl.NumberFormat('en-IN').format(Math.round(data.predicted_price_inr))}`;
                resultBox.style.display = 'block';
            });
        </script>
    </body>
    </html>
    """


@app.post('/api/predict')
def predict_price(payload: PredictionInput):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail='Model could not be loaded.')

    input_df = pd.DataFrame([payload.model_dump()])
    predicted_value = float(model.predict(input_df)[0])
    lower_bound = predicted_value * 0.9
    upper_bound = predicted_value * 1.1

    return {
        'predicted_price_inr': int(round(predicted_value)),
        'range_inr': {
            'min': int(round(lower_bound)),
            'max': int(round(upper_bound)),
        },
    }
