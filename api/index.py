from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / 'used_phones.csv'
CATALOG_FILE = BASE_DIR / 'device_catalog.json'
MODEL_FILE = Path(tempfile.gettempdir()) / 'used_phone_price_model.joblib'


class PredictionInput(BaseModel):
    brand: str = Field(..., min_length=1)
    model_name: str = Field(default='')
    country_code: str = Field(default='IN', min_length=2, max_length=2)
    price_mode: Literal['market', 'custom'] = 'custom'
    custom_price: float | None = Field(default=None, gt=0)
    original_price_inr: int | None = Field(default=None, gt=0)
    ram_gb: int = Field(..., ge=4, le=16)
    storage_gb: int = Field(..., ge=64, le=512)
    has_5g: int = Field(..., ge=0, le=1)
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


@lru_cache(maxsize=1)
def load_catalog():
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(f'Device catalog not found: {CATALOG_FILE}')
    return json.loads(CATALOG_FILE.read_text(encoding='utf-8'))


def find_country(country_code: str):
    for country in load_catalog()['countries']:
        if country['code'] == country_code.upper():
            return country
    raise HTTPException(status_code=422, detail='Select a supported country.')


def find_device(brand_name: str, model_name: str):
    for brand in load_catalog()['brands']:
        if brand['name'].casefold() != brand_name.casefold():
            continue
        for model in brand['models']:
            if model['name'].casefold() == model_name.casefold():
                return brand, model
    return None, None


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


@app.get('/api/catalog')
def get_device_catalog():
    return load_catalog()


@app.get('/robots.txt', response_class=PlainTextResponse)
def robots():
    return """User-agent: *
Allow: /
Disallow: /api/

Sitemap: https://pricepredictions.vercel.app/sitemap.xml
"""


@app.get('/sitemap.xml', response_class=PlainTextResponse)
def sitemap():
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://pricepredictions.vercel.app/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
"""


@app.get('/', response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta name="theme-color" content="#f5f7f5" />
      <meta name="description" content="Estimate your used phone's resale value in seconds. Compare popular mobile brands and models, choose a country, and get a personalized price estimate." />
      <meta name="robots" content="index, follow, max-image-preview:large" />
      <meta name="keywords" content="used phone price, mobile resale value, phone resale calculator, used mobile price predictor, smartphone valuation, iPhone resale price, Samsung resale price" />
      <link rel="canonical" href="https://pricepredictions.vercel.app/" />
      <meta property="og:type" content="website" />
      <meta property="og:site_name" content="Revalue" />
      <meta property="og:title" content="Used Phone Resale Price Calculator | Revalue" />
      <meta property="og:description" content="Find an estimated resale value for your used phone. Choose from popular brands and models or enter a custom device and price." />
      <meta property="og:url" content="https://pricepredictions.vercel.app/" />
      <meta name="twitter:card" content="summary" />
      <meta name="twitter:title" content="Used Phone Resale Price Calculator | Revalue" />
      <meta name="twitter:description" content="Estimate your used smartphone's resale value with Revalue's phone price calculator." />
      <link rel="icon" href="data:," />
      <title>Used Phone Resale Price Calculator | Revalue</title>
      <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "WebApplication",
          "name": "Revalue Used Phone Resale Price Calculator",
          "url": "https://pricepredictions.vercel.app/",
          "description": "Estimate the resale value of a used smartphone by brand, model, original price, age, and condition.",
          "applicationCategory": "UtilitiesApplication",
          "operatingSystem": "Any",
          "browserRequirements": "Requires JavaScript"
        }
      </script>
      <style>
        :root {
          color-scheme: light;
          --ink: #152321;
          --muted: #667571;
          --line: #e5ebe7;
          --surface: #ffffff;
          --canvas: #f5f7f5;
          --green: #145d4b;
          --green-dark: #104a3c;
          --mint: #e9f5ef;
          --lime: #c9ef76;
          --shadow: 0 20px 55px rgba(31, 57, 48, .08);
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          color: var(--ink);
          background: var(--canvas);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          -webkit-font-smoothing: antialiased;
        }
        button, input, select { font: inherit; }
        .topbar {
          height: 76px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          max-width: 1240px;
          padding: 0 28px;
          margin: 0 auto;
        }
        .brandmark { display:flex; align-items:center; gap:11px; font-weight:750; letter-spacing:-.04em; font-size:21px; }
        .brand-icon { display:grid; place-items:center; width:36px; height:36px; border-radius:12px; color:white; background:var(--green); font-size:19px; }
        .top-actions { display:flex; align-items:center; gap:16px; color:var(--muted); font-size:13px; }
        .country-wrap { display:flex; align-items:center; gap:8px; }
        .country-wrap select { border:0; background:transparent; color:var(--ink); font-weight:650; padding:8px 20px 8px 4px; outline-offset:3px; }
        .top-divider { height:24px; border-left:1px solid var(--line); }
        .top-note { color:var(--muted); }
        main { max-width:1240px; padding:34px 28px 64px; margin:0 auto; }
        .hero { display:flex; justify-content:space-between; align-items:flex-end; gap:28px; margin-bottom:30px; }
        .eyebrow { display:inline-flex; align-items:center; gap:8px; font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--green); font-weight:750; }
        .eyebrow:before { content:""; width:7px; height:7px; border-radius:50%; background:#8dbb45; }
        h1 { margin:13px 0 10px; max-width:700px; font-size:clamp(34px,4vw,52px); line-height:1.04; letter-spacing:-.055em; font-weight:740; }
        .hero p { max-width:650px; margin:0; color:var(--muted); font-size:15px; line-height:1.7; }
        .trust-chip { display:inline-flex; gap:8px; align-items:center; padding:10px 13px; color:#47635b; background:#edf2ee; border:1px solid #e3eae5; border-radius:999px; font-size:12px; font-weight:650; white-space:nowrap; }
        .trust-chip span { color:#52875c; font-size:15px; }
        .layout { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(300px,.82fr); align-items:start; gap:22px; }
        .card { background:var(--surface); border:1px solid #e9eeeb; border-radius:20px; box-shadow:var(--shadow); }
        .form-card { padding:28px; }
        .section-heading { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:23px; }
        .section-heading h2 { margin:0 0 6px; font-size:19px; letter-spacing:-.035em; }
        .section-heading p { margin:0; color:var(--muted); font-size:13px; line-height:1.5; }
        .step-label { color:#718079; font-size:11px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
        .form-section + .form-section { margin-top:25px; padding-top:24px; border-top:1px solid var(--line); }
        .section-title { display:flex; align-items:center; gap:10px; margin-bottom:17px; font-size:14px; font-weight:730; letter-spacing:-.01em; }
        .number { display:grid; place-items:center; width:25px; height:25px; border-radius:8px; background:var(--mint); color:var(--green); font-size:12px; }
        .fields { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
        .field { min-width:0; }
        .field.full { grid-column:1/-1; }
        .field label { display:flex; justify-content:space-between; align-items:center; margin-bottom:7px; color:#364640; font-size:12px; font-weight:650; }
        .hint { color:#87938e; font-size:11px; font-weight:450; }
        .control {
          width:100%; min-height:46px; border:1px solid #dce5df; border-radius:10px;
          background:#fff; color:var(--ink); padding:0 13px; outline:none; transition:.16s ease;
        }
        .control:focus { border-color:#68a88e; box-shadow:0 0 0 3px rgba(54,137,105,.13); }
        .control::placeholder { color:#a0aaa5; }
        .price-modes { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
        .price-mode {
          position:relative; display:flex; align-items:flex-start; gap:10px; min-height:76px;
          padding:14px; border:1px solid #dce5df; border-radius:12px; cursor:pointer;
          transition:.16s ease;
        }
        .price-mode:has(input:checked) { border-color:#337c63; background:#f1f8f3; box-shadow:inset 0 0 0 1px #337c63; }
        .price-mode input { margin:3px 0 0; accent-color:var(--green); }
        .price-mode strong { display:block; font-size:12px; line-height:1.35; }
        .price-mode small { display:block; margin-top:4px; color:var(--muted); font-size:11px; line-height:1.4; }
        .price-preview { margin-top:11px; padding:11px 13px; display:flex; justify-content:space-between; align-items:center; border:1px dashed #d4e2d8; border-radius:10px; background:#fbfdfb; color:var(--muted); font-size:12px; }
        .price-preview strong { color:var(--green); font-size:15px; }
        .price-disclaimer { margin:9px 0 0; color:#89958f; font-size:10px; line-height:1.5; }
        .range-row { display:flex; align-items:center; gap:10px; }
        .range-row input[type="range"] { width:100%; accent-color:var(--green); }
        .range-value { min-width:45px; text-align:right; font-size:13px; font-weight:700; }
        .condition-help { margin:11px 0 0; color:#85918b; font-size:11px; line-height:1.5; }
        .check-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
        .check-card { display:flex; align-items:center; gap:9px; padding:12px; border:1px solid var(--line); border-radius:10px; color:#43524c; font-size:12px; }
        .check-card input { accent-color:var(--green); width:15px; height:15px; }
        .submit {
          width:100%; min-height:52px; display:flex; justify-content:center; align-items:center; gap:10px;
          border:0; border-radius:12px; margin-top:24px; color:#fff; background:var(--green);
          font-weight:730; cursor:pointer; transition:.16s ease;
        }
        .submit:hover { background:var(--green-dark); transform:translateY(-1px); }
        .submit:disabled { opacity:.65; cursor:wait; transform:none; }
        .result-card { position:sticky; top:20px; overflow:hidden; }
        .result-top { padding:25px; background:linear-gradient(145deg,#164f40,#103d34); color:white; }
        .result-kicker { color:#c0dacd; font-size:11px; font-weight:700; letter-spacing:.11em; text-transform:uppercase; }
        .result-title { margin:10px 0 7px; font-size:22px; letter-spacing:-.04em; }
        .result-sub { margin:0; max-width:310px; color:#c6d8d0; font-size:12px; line-height:1.6; }
        .result-value { display:flex; align-items:flex-start; gap:8px; margin:29px 0 4px; font-size:42px; font-weight:760; letter-spacing:-.06em; line-height:1; }
        .result-value .currency { margin-top:5px; font-size:18px; letter-spacing:0; color:#d0e3d8; }
        .result-range { color:#d1e1d9; font-size:12px; }
        .result-body { padding:21px 25px 24px; }
        .result-placeholder { padding:18px 14px; text-align:center; border:1px dashed #dfe8e2; border-radius:13px; background:#fafcf9; }
        .phone-art { position:relative; width:55px; height:82px; margin:0 auto 14px; border:3px solid #d8e3dc; border-radius:13px; background:linear-gradient(140deg,#f3f8f4,#e8f1eb); }
        .phone-art:before { content:""; position:absolute; width:13px; height:3px; top:5px; left:18px; border-radius:4px; background:#d0ddd4; }
        .phone-art:after { content:""; position:absolute; width:22px; height:22px; right:5px; top:12px; border-radius:8px; background:#dbece0; }
        .result-placeholder strong { display:block; font-size:13px; }
        .result-placeholder p { margin:5px 0 0; color:var(--muted); font-size:11px; line-height:1.55; }
        .result-details { display:none; }
        .detail-row { display:flex; justify-content:space-between; gap:12px; padding:11px 0; border-bottom:1px solid #edf1ee; font-size:12px; }
        .detail-row span:first-child { color:var(--muted); }
        .detail-row strong { text-align:right; }
        .disclaimer { display:flex; gap:9px; margin-top:15px; padding:12px; border-radius:10px; background:#f3f6f2; color:#68766f; font-size:10px; line-height:1.55; }
        .disclaimer b { color:#43544c; }
        .error { display:none; margin-top:13px; padding:11px 13px; border:1px solid #f1d0ca; border-radius:10px; background:#fff6f4; color:#9d382a; font-size:12px; line-height:1.5; }
        .footer-note { margin:22px 0 0; color:#8a9690; text-align:center; font-size:11px; }
        @media (max-width:900px) {
          main { padding-top:23px; }
          .layout { grid-template-columns:1fr; }
          .result-card { position:static; }
          .hero { align-items:flex-start; flex-direction:column; }
        }
        @media (max-width:560px) {
          .topbar { height:65px; padding:0 18px; }
          .top-note,.top-divider { display:none; }
          main { padding:20px 14px 42px; }
          .hero { margin-bottom:22px; gap:16px; }
          h1 { font-size:36px; }
          .hero p { font-size:13px; }
          .trust-chip { font-size:10px; }
          .form-card { padding:20px 16px; border-radius:16px; }
          .fields { gap:13px 10px; }
          .price-modes { grid-template-columns:1fr; }
          .check-row { grid-template-columns:1fr; }
          .result-top { padding:22px; }
          .result-body { padding:18px 18px 21px; }
        }
      </style>
    </head>
    <body>
      <header class="topbar">
        <div class="brandmark"><span class="brand-icon" aria-hidden="true">↗</span> revalue</div>
        <div class="top-actions">
          <span class="top-note">A smarter way to sell your phone</span>
          <span class="top-divider" aria-hidden="true"></span>
          <label class="country-wrap" for="country">🌐 <select id="country" aria-label="Choose country"></select></label>
        </div>
      </header>
      <main>
        <section class="hero">
          <div>
            <span class="eyebrow">Phone resale estimator</span>
            <h1>Your phone still has<br />value. Find out how much.</h1>
            <p>Choose your device, tell us a little about its condition, and get a resale estimate in seconds.</p>
          </div>
          <div class="trust-chip"><span aria-hidden="true">✳</span> Free estimate · No sign-up required</div>
        </section>
        <div class="layout">
          <form class="card form-card" id="predictForm">
            <div class="section-heading">
              <div><h2>Tell us about your phone</h2><p>A few details help us make your estimate more useful.</p></div>
              <span class="step-label">01 / 03</span>
            </div>
            <section class="form-section">
              <div class="section-title"><span class="number">1</span> Device details</div>
              <div class="fields">
                <div class="field">
                  <label for="brand">Brand</label>
                  <select class="control" id="brand" required></select>
                  <input class="control" id="customBrand" placeholder="Enter brand name" hidden />
                </div>
                <div class="field">
                  <label for="model">Model</label>
                  <select class="control" id="model" required></select>
                  <input class="control" id="customModel" placeholder="Enter model name" hidden />
                </div>
                <div class="field">
                  <label for="ram_gb">RAM <span class="hint">GB</span></label>
                  <select class="control" id="ram_gb"><option>4</option><option>6</option><option selected>8</option><option>12</option><option>16</option></select>
                </div>
                <div class="field">
                  <label for="storage_gb">Storage <span class="hint">GB</span></label>
                  <select class="control" id="storage_gb"><option>64</option><option selected>128</option><option>256</option><option>512</option></select>
                </div>
              </div>
            </section>
            <section class="form-section">
              <div class="section-title"><span class="number">2</span> Price reference</div>
              <div class="price-modes">
                <label class="price-mode"><input type="radio" name="priceMode" value="market" checked /><span><strong>Use estimated market price</strong><small>Start with our model reference for this device</small></span></label>
                <label class="price-mode"><input type="radio" name="priceMode" value="custom" /><span><strong>Enter a custom price</strong><small>Use the amount you paid or prefer</small></span></label>
              </div>
              <div class="price-preview" id="marketPreview"><span>Estimated reference price</span><strong id="marketPrice">—</strong></div>
              <div class="field" id="customPriceField" style="display:none;margin-top:12px">
                <label for="customPrice">Your price <span class="hint" id="priceCurrency">INR</span></label>
                <input class="control" id="customPrice" type="number" min="1" step="1" placeholder="Enter a price" />
              </div>
              <p class="price-disclaimer" id="priceDisclaimer">Reference prices are curated estimates converted from INR—not live retailer listings. Choose a country above for display currency.</p>
            </section>
            <section class="form-section">
              <div class="section-title"><span class="number">3</span> Age &amp; condition</div>
              <div class="fields">
                <div class="field">
                  <label for="months_used">Time used <span class="hint">months</span></label>
                  <input class="control" id="months_used" type="number" min="1" max="47" value="12" required />
                </div>
                <div class="field">
                  <label for="battery_health_pct">Battery health <span class="range-value" id="batteryValue">85%</span></label>
                  <div class="range-row"><input id="battery_health_pct" type="range" min="65" max="100" value="85" /><span class="hint">65–100%</span></div>
                </div>
                <div class="field">
                  <label for="screen_condition">Screen condition <span class="range-value" id="screenValue">4 / 5</span></label>
                  <div class="range-row"><input id="screen_condition" type="range" min="1" max="5" value="4" /><span class="hint">1–5</span></div>
                </div>
                <div class="field">
                  <label for="body_condition">Body condition <span class="range-value" id="bodyValue">4 / 5</span></label>
                  <div class="range-row"><input id="body_condition" type="range" min="1" max="5" value="4" /><span class="hint">1–5</span></div>
                </div>
                <div class="field full">
                  <div class="check-row">
                    <label class="check-card"><input id="has_5g" type="checkbox" checked /> 5G enabled</label>
                    <label class="check-card"><input id="under_warranty" type="checkbox" /> Still under warranty</label>
                    <label class="check-card"><input id="has_box_bill" type="checkbox" checked /> Box and bill available</label>
                  </div>
                  <p class="condition-help">Condition scale: 1 = heavily worn · 5 = like new. Be honest for the most helpful estimate.</p>
                </div>
              </div>
            </section>
            <button class="submit" id="submitBtn" type="submit"><span>Calculate my estimate</span><span aria-hidden="true">→</span></button>
            <div class="error" id="errorMessage" role="alert"></div>
          </form>
          <aside class="card result-card" aria-live="polite">
            <div class="result-top">
              <div class="result-kicker">Your estimate</div>
              <h2 class="result-title" id="resultTitle">Ready when you are</h2>
              <p class="result-sub" id="resultSubtitle">Complete your phone details to see an estimated resale value and a helpful price range.</p>
              <div class="result-value" id="resultValue" hidden><span class="currency" id="resultSymbol">₹</span><span id="predictedValue">—</span></div>
              <div class="result-range" id="resultRange" hidden></div>
            </div>
            <div class="result-body">
              <div class="result-placeholder" id="resultPlaceholder">
                <div class="phone-art" aria-hidden="true"></div>
                <strong>Your estimate will appear here</strong>
                <p>It takes about 30 seconds. Your details stay private.</p>
              </div>
              <div class="result-details" id="resultDetails">
                <div class="detail-row"><span>Device</span><strong id="detailDevice">—</strong></div>
                <div class="detail-row"><span>Price basis</span><strong id="detailPriceBasis">—</strong></div>
                <div class="detail-row"><span>Condition</span><strong id="detailCondition">—</strong></div>
              </div>
              <div class="disclaimer"><span aria-hidden="true">ⓘ</span><span><b>For guidance only.</b> This estimate uses synthetic training data and curated reference prices. It is not a guaranteed offer or live marketplace quote.</span></div>
            </div>
          </aside>
        </div>
        <p class="footer-note">Made for smarter, more confident resale decisions.</p>
      </main>
      <script>
        const $ = (id) => document.getElementById(id);
        let catalog = null;
        const customBrandOption = 'Other / custom brand';
        const customModelOption = 'Other / enter model';

        function selectedCountry() {
          return catalog.countries.find((item) => item.code === $('country').value);
        }
        function formatMoney(amount, country = selectedCountry()) {
          return new Intl.NumberFormat(country.locale, {
            style: 'currency', currency: country.currency, maximumFractionDigits: 0
          }).format(Math.round(amount));
        }
        function selectedBrand() {
          return catalog.brands.find((item) => item.name === $('brand').value);
        }
        function selectedModel() {
          const brand = selectedBrand();
          return brand?.models.find((item) => item.name === $('model').value);
        }
        function isCustomDevice() {
          return $('brand').value === customBrandOption || $('model').value === customModelOption;
        }
        function refreshPrice() {
          if (!catalog) return;
          const country = selectedCountry();
          const model = selectedModel();
          const currencyMark = `${country.currency} · ${country.symbol}`;
          $('priceCurrency').textContent = currencyMark;
          $('customPrice').placeholder = `Enter amount in ${country.currency}`;
          const useCustom = document.querySelector('input[name="priceMode"]:checked').value === 'custom';
          $('customPriceField').style.display = useCustom ? 'block' : 'none';
          $('marketPreview').style.display = useCustom ? 'none' : 'flex';
          if (!useCustom && model) {
            $('marketPrice').textContent = formatMoney(model.reference_price_inr * country.rate_from_inr);
          } else if (!useCustom) {
            $('marketPrice').textContent = 'Not listed';
          }
          const autoMode = document.querySelector('input[name="priceMode"][value="market"]');
          autoMode.disabled = isCustomDevice();
          if (isCustomDevice() && !useCustom) {
            document.querySelector('input[name="priceMode"][value="custom"]').checked = true;
            $('customPriceField').style.display = 'block';
            $('marketPreview').style.display = 'none';
          }
        }
        function refreshModels() {
          const brand = selectedBrand();
          $('model').replaceChildren();
          if (brand) {
            for (const item of brand.models) {
              $('model').add(new Option(item.name, item.name));
            }
          }
          $('model').add(new Option(customModelOption, customModelOption));
          $('customBrand').hidden = $('brand').value !== customBrandOption;
          $('customModel').hidden = $('model').value !== customModelOption;
          refreshPrice();
        }
        async function initializeCatalog() {
          const response = await fetch('/api/catalog');
          if (!response.ok) throw new Error('Could not load the phone catalog. Please refresh and try again.');
          catalog = await response.json();
          for (const country of catalog.countries) {
            $('country').add(new Option(`${country.name} (${country.currency})`, country.code));
          }
          $('country').value = 'IN';
          for (const brand of catalog.brands) $('brand').add(new Option(brand.name, brand.name));
          $('brand').add(new Option(customBrandOption, customBrandOption));
          $('brand').value = 'Apple';
          refreshModels();
          refreshPrice();
        }

        $('brand').addEventListener('change', refreshModels);
        $('model').addEventListener('change', () => {
          const custom = $('model').value === customModelOption;
          $('customModel').hidden = !custom;
          refreshPrice();
        });
        $('country').addEventListener('change', refreshPrice);
        document.querySelectorAll('input[name="priceMode"]').forEach((input) => input.addEventListener('change', refreshPrice));
        $('battery_health_pct').addEventListener('input', () => $('batteryValue').textContent = `${$('battery_health_pct').value}%`);
        $('screen_condition').addEventListener('input', () => $('screenValue').textContent = `${$('screen_condition').value} / 5`);
        $('body_condition').addEventListener('input', () => $('bodyValue').textContent = `${$('body_condition').value} / 5`);
        initializeCatalog().catch((error) => {
          $('errorMessage').textContent = error.message;
          $('errorMessage').style.display = 'block';
        });

        $('predictForm').addEventListener('submit', async (event) => {
          event.preventDefault();
          const errorBox = $('errorMessage');
          errorBox.style.display = 'none';
          const button = $('submitBtn');
          button.disabled = true;
          button.querySelector('span').textContent = 'Calculating…';
          const customBrand = $('brand').value === customBrandOption;
          const customModel = $('model').value === customModelOption;
          const brand = customBrand ? $('customBrand').value.trim() : $('brand').value;
          const model = customModel ? $('customModel').value.trim() : $('model').value;
          const priceMode = document.querySelector('input[name="priceMode"]:checked').value;
          if (!brand || !model) {
            errorBox.textContent = 'Enter both the brand and model name to continue.';
            errorBox.style.display = 'block';
            button.disabled = false;
            button.querySelector('span').textContent = 'Calculate my estimate';
            return;
          }
          const payload = {
            brand, model_name: model, country_code: $('country').value, price_mode: priceMode,
            custom_price: priceMode === 'custom' ? Number($('customPrice').value) : null,
            ram_gb: Number($('ram_gb').value), storage_gb: Number($('storage_gb').value),
            has_5g: Number($('has_5g').checked), months_used: Number($('months_used').value),
            screen_condition: Number($('screen_condition').value), body_condition: Number($('body_condition').value),
            battery_health_pct: Number($('battery_health_pct').value),
            under_warranty: Number($('under_warranty').checked), has_box_bill: Number($('has_box_bill').checked)
          };
          try {
            const response = await fetch('/api/predict', {
              method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'We could not calculate an estimate. Check your details and try again.');
            $('resultPlaceholder').style.display = 'none';
            $('resultDetails').style.display = 'block';
            $('resultValue').hidden = false;
            $('resultRange').hidden = false;
            $('resultTitle').textContent = 'Estimated resale value';
            $('resultSubtitle').textContent = 'A helpful starting point for your next sale.';
            $('resultSymbol').textContent = selectedCountry().symbol;
            $('predictedValue').textContent = new Intl.NumberFormat(selectedCountry().locale, {maximumFractionDigits:0}).format(data.predicted_price);
            $('resultRange').textContent = `Likely range: ${formatMoney(data.range.min)} – ${formatMoney(data.range.max)}`;
            $('detailDevice').textContent = `${brand} ${model}`;
            $('detailPriceBasis').textContent = priceMode === 'market' ? 'Curated reference' : `Custom · ${formatMoney(payload.custom_price)}`;
            $('detailCondition').textContent = `${payload.months_used} months · ${payload.battery_health_pct}% battery`;
          } catch (error) {
            errorBox.textContent = error.message;
            errorBox.style.display = 'block';
          } finally {
            button.disabled = false;
            button.querySelector('span').textContent = 'Calculate my estimate';
          }
        });
      </script>
      <script src="https://mcp.figma.com/mcp/html-to-design/capture.js" async></script>
    </body>
    </html>
    """


@app.post('/api/predict')
def predict_price(payload: PredictionInput):
    country = find_country(payload.country_code)
    catalog_brand, catalog_model = find_device(payload.brand, payload.model_name)

    if payload.price_mode == 'market':
        if catalog_model is None:
            raise HTTPException(
                status_code=422,
                detail='Automatic reference pricing is unavailable for this model. Enter a custom price instead.',
            )
        original_price_inr = int(catalog_model['reference_price_inr'])
        price_source = 'curated_reference'
    elif payload.custom_price is not None:
        original_price_inr = round(payload.custom_price / country['rate_from_inr'])
        price_source = 'custom'
    elif payload.original_price_inr is not None:
        original_price_inr = payload.original_price_inr
        price_source = 'custom'
    else:
        raise HTTPException(
            status_code=422,
            detail='Enter a positive custom price or choose a listed model for automatic reference pricing.',
        )

    model = get_model()
    input_df = pd.DataFrame([{
        'brand': catalog_brand['name'] if catalog_brand else payload.brand,
        'ram_gb': payload.ram_gb,
        'storage_gb': payload.storage_gb,
        'has_5g': payload.has_5g,
        'original_price_inr': original_price_inr,
        'months_used': payload.months_used,
        'screen_condition': payload.screen_condition,
        'body_condition': payload.body_condition,
        'battery_health_pct': payload.battery_health_pct,
        'under_warranty': payload.under_warranty,
        'has_box_bill': payload.has_box_bill,
    }])
    predicted_value = float(model.predict(input_df)[0])
    lower_bound = predicted_value * 0.9
    upper_bound = predicted_value * 1.1

    return {
        'predicted_price_inr': int(round(predicted_value)),
        'range_inr': {
            'min': int(round(lower_bound)),
            'max': int(round(upper_bound)),
        },
        'predicted_price': int(round(predicted_value * country['rate_from_inr'])),
        'range': {
            'min': int(round(lower_bound * country['rate_from_inr'])),
            'max': int(round(upper_bound * country['rate_from_inr'])),
        },
        'country': country['code'],
        'currency': country['currency'],
        'price_source': price_source,
    }
