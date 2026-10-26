# Used Mobile Resale Price Predictor (Revalue)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-pricepredictions.vercel.app-145d4b?style=for-the-badge&logo=vercel)](https://pricepredictions.vercel.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Revalue** is an end-to-end Machine Learning web application that predicts the realistic secondary resale valuation of used mobile phones and smartphones. It pairs realistic market depreciation modeling with multi-currency support and an interactive web interface.

🌐 **Live Deployed Application:** [https://pricepredictions.vercel.app/](https://pricepredictions.vercel.app/)

Created by: **Archit Bishnoi**

---

## 🌟 Key Features

- **Searchable Device Catalog**: Supports **120+ popular mobile phone models across 25 brands** (Apple, Samsung, Xiaomi, OnePlus, Google, vivo, realme, Nothing, Motorola, Sony, POCO, Redmi, etc.), plus support for custom brands and models.
- **Multi-Country & Multi-Currency**: Real-time conversion across major international currencies (INR `₹`, USD `$`, EUR `€`, GBP `£`, AED `AED`, CAD `CA$`, AUD `A$`, etc.).
- **Dual Valuation Mode**: Use curated market reference prices or specify custom original purchase invoices.
- **Accurate Attribute Modeling**:
  - **5G Connectivity**: 5G-ready devices retain significantly higher secondary demand and market premium.
  - **Warranty Coverage**: Realistic non-linear appreciation for active standard and extended warranties.
  - **Original Packaging & Bill**: Premium resale valuation retention when box and invoice accessories are intact.
  - **Physical Condition & Battery Health**: Fine-grained scoring for screen condition, body scuffs/dents, and battery wear.
  - **Age Depreciation Curve**: Realistic exponential decay model reflecting true secondary smartphone depreciation.
- **Dual Application Architecture**:
  - **FastAPI + Modern Web UI** (Serverless production build deployed on Vercel).
  - **Streamlit Web App** (Local development & data exploration tool).
- **SEO & Social Sharing Ready**: Fully structured JSON-LD schema, canonical tags, OpenGraph previews, `sitemap.xml`, and `robots.txt`.

---

## 🏗️ Architecture & How It Works

```
                     ┌─────────────────────────────┐
                     │     device_catalog.json     │
                     │  (25 Brands, 120+ Models)   │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
┌───────────────────────────┐  Training   ┌───────────────────────────────┐
│    generate_dataset.py    │ ──────────> │     RandomForestRegressor     │
│ (8,000 Balanced Samples)  │             │   (Pipeline + OneHotEncoder)  │
└───────────────────────────┘             └───────────────┬───────────────┘
                                                          │
                        ┌─────────────────────────────────┴─────────────────────────────────┐
                        ▼                                                                   ▼
         ┌───────────────────────────────┐                                   ┌───────────────────────────────┐
         │     FastAPI Production App    │                                   │       Streamlit Local App     │
         │         (api/index.py)        │                                   │           (app.py)            │
         │  Live: pricepredictions.app   │                                   │   Interactive ML Dashboard   │
         └───────────────────────────────┘                                   └───────────────────────────────┘
```

1. **Synthetic Data Engine (`generate_dataset.py`)**:
   - Generates 8,000 realistic records spanning all 25 catalog brands.
   - Applies an exponential decay formula `exp(-0.028 * months_used)` with multipliers for brand retention base, screen & body condition, battery health degradation, 5G premium (`1.10x`), warranty bonus (`1.12x`), and box/bill availability (`1.10x`).
2. **Machine Learning Pipeline**:
   - Features: `brand`, `ram_gb`, `storage_gb`, `has_5g`, `original_price_inr`, `months_used`, `screen_condition`, `body_condition`, `battery_health_pct`, `under_warranty`, `has_box_bill`.
   - Preprocessing: `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` for brands and passthrough for numerical dimensions.
   - Regressor: `RandomForestRegressor(n_estimators=300, random_state=42)` yielding strong $R^2 \approx 0.96$ and low MAE.
3. **Serving Layer**:
   - **Production (Vercel)**: FastAPI handles `/api/predict` (POST) and `/api/catalog` (GET) along with embedded responsive modern CSS/JS UI and SEO endpoints (`robots.txt`, `sitemap.xml`).
   - **Local Studio**: Streamlit app for real-time slider experimentation and dataset inspection.

---

## 📁 Project Structure

```
used_mobile_price_prediction/
├── api/
│   ├── index.py                  # FastAPI serverless entrypoint for Vercel
│   └── __init__.py
├── app.py                        # Streamlit web application
├── device_catalog.json           # Catalog of 25 brands, 120+ devices & exchange rates
├── generate_dataset.py           # Synthetic dataset generator script
├── used_phones.csv               # 8,000 balanced phone valuation records
├── used_accessories.csv          # Companion dataset for mobile accessories
├── requirements.txt              # Production Python package dependencies
├── vercel.json                   # Vercel deployment configuration & routing
└── README.md                     # Project documentation
```

---

## 🛠️ Step-by-Step Cloning & Setup Guide

Want to run this project on your own machine or deploy your own copy of the website? Follow these easy steps:

### Prerequisites
Make sure you have installed:
- [Git](https://git-scm.com/)
- [Python 3.10+](https://www.python.org/downloads/)

---

### Step 1: Fork & Clone the Repository
1. On GitHub, fork the repository by clicking **Fork** at the top right of [Price_Predictions_of_used_mobiles](https://github.com/Archit0029/Price_Predictions_of_used_mobiles).
2. Open your terminal or Command Prompt and clone your copy:
```bash
git clone https://github.com/Archit0029/Price_Predictions_of_used_mobiles.git
cd Price_Predictions_of_used_mobiles
```

---

### Step 2: Create and Activate a Virtual Environment
It is recommended to use an isolated Python environment:

- **On Windows (PowerShell / CMD):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

### Step 3: Install Required Dependencies
Install all required libraries including FastAPI, Uvicorn, Streamlit, Pandas, and scikit-learn:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: (Optional) Re-generate Dataset
The repository already includes the pre-trained dataset `used_phones.csv`. If you want to customize sample size or tweak depreciation weights:
```bash
python generate_dataset.py --phones 8000 --accessories 1500 --output-dir .
```

---

### Step 5: Run the Project Locally

You have **two interfaces** to choose from:

#### 🔹 Option A: Run the Live Website UI (FastAPI Server)
This runs the exact same UI and API that is deployed in production:
```bash
uvicorn api.index:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

#### 🔹 Option B: Run the Streamlit Dashboard
If you prefer an interactive data-science slider dashboard with training data preview:
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

### Step 6: Deploy Your Own Copy to Vercel (Free 1-Click Setup)

Deploying your cloned website to the internet takes less than 2 minutes:

1. Push your cloned repository to your GitHub account.
2. Sign up or log in at **[Vercel](https://vercel.com/)**.
3. Click **Add New...** ➔ **Project** and select your GitHub repository.
4. Keep the default settings (Framework preset: `Other`, Root Directory: `./`).
5. Click **Deploy**.
6. Vercel will automatically read `vercel.json` and `requirements.txt` and launch your live site with an official `.vercel.app` domain!

---

## 🌐 Live Deployment on Vercel

The live official web application is hosted at:  
🔗 **[https://pricepredictions.vercel.app/](https://pricepredictions.vercel.app/)**

### Deployment Details:
- **Platform**: Vercel Serverless Functions (Python runtime via `vercel.json`).
- **REST API Endpoints**:
  - `GET /` — Responsive web application UI.
  - `GET /api/catalog` — Returns available countries, currency rates, brands, and device models.
  - `POST /api/predict` — Receives phone condition attributes and outputs estimated valuation and ranges.
  - `GET /robots.txt` & `GET /sitemap.xml` — Search engine discovery routes.

---

## 📊 Model Evaluation Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **$R^2$ Score** | **~0.96** | High proportion of price variance explained by features |
| **Mean Absolute Error (MAE)** | **~₹1,600** | Tight prediction boundary relative to high-tier device costs |
| **Monotonic Consistency** | **Verified** | 5G, warranty, and box/bill options strictly increase resale value |

---

## 🤝 Contributing

Contributions, issues, and feature suggestions are always welcome!
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License & Attribution

Developed with ❤️ by **Archit Bishnoi**.  
*Disclaimer: Reference prices and synthetic training data are curated for estimation and educational modeling purposes and do not represent live binding retail quotes.*
