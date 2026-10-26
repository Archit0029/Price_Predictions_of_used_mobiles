import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / 'used_phones.csv'
CATALOG_FILE = PROJECT_DIR / 'device_catalog.json'
MODEL_FILE = PROJECT_DIR / 'used_phone_price_model.joblib'
CUSTOM_BRAND = 'Other / custom brand'
CUSTOM_MODEL = 'Other / enter model'


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_catalog():
    return json.loads(CATALOG_FILE.read_text(encoding='utf-8'))


@st.cache_data
def build_model_and_metrics():
    df = load_data()
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
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
    }
    return model, metrics


st.set_page_config(
    page_title='Revalue — Phone resale estimate',
    page_icon='📱',
    layout='wide',
    menu_items={'Get help': None, 'Report a bug': None, 'About': None},
)

st.markdown(
    """
    <style>
    div[data-testid="stHeaderActionElements"], div[data-testid="stHeader"] {
        display: none !important;
    }
    .block-container {max-width: 1240px; padding-top: 2rem;}
    .stButton > button[kind="primary"] {
        background: #145d4b; border: 0; border-radius: 10px;
        min-height: 48px; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not DATA_FILE.exists() or not CATALOG_FILE.exists():
    st.error('The phone dataset or device catalog is missing from the project folder.')
    st.stop()

catalog = load_catalog()
model, model_metrics = build_model_and_metrics()

st.markdown('### ↗ revalue')
st.caption('PHONE RESALE ESTIMATOR')
st.title("Your phone still has value. Find out how much.")
st.write('Choose your device, tell us a little about its condition, and get a resale estimate in seconds.')

with st.sidebar:
    st.subheader('Country and currency')
    country_names = {item['name']: item for item in catalog['countries']}
    country_name = st.selectbox('Choose country', list(country_names), index=0)
    country = country_names[country_name]
    st.caption(f'Prices are shown in {country["currency"]}.')
    st.divider()
    st.subheader('Model quality')
    st.metric('Mean absolute error', f'{country["symbol"]}{model_metrics["MAE"] * country["rate_from_inr"]:,.0f}')
    st.metric('R² score', f'{model_metrics["R2"]:.3f}')

device_col, condition_col = st.columns([1.05, 0.95], gap='large')

with device_col:
    st.subheader('1. Device details')
    brand_names = [item['name'] for item in catalog['brands']] + [CUSTOM_BRAND]
    selected_brand = st.selectbox('Brand', brand_names)
    brand_entry = next(
        (item for item in catalog['brands'] if item['name'] == selected_brand),
        None,
    )
    brand = selected_brand
    if brand_entry is None:
        brand = st.text_input('Enter brand name', placeholder='e.g. Fairphone')
        models = []
    else:
        models = brand_entry['models']

    model_names = [item['name'] for item in models] + [CUSTOM_MODEL]
    selected_model = st.selectbox('Model', model_names)
    model_entry = next(
        (item for item in models if item['name'] == selected_model),
        None,
    )
    model_name = selected_model
    if model_entry is None:
        model_name = st.text_input('Enter model name', placeholder='e.g. Phone model')

    ram_gb = st.selectbox('RAM (GB)', [4, 6, 8, 12, 16], index=2)
    storage_gb = st.selectbox('Storage (GB)', [64, 128, 256, 512], index=1)

    st.subheader('2. Price reference')
    price_mode = st.radio(
        'Choose a price option',
        ['Use estimated market price', 'Enter a custom price'],
        horizontal=True,
    )
    if price_mode == 'Use estimated market price' and model_entry:
        original_price_inr = model_entry['reference_price_inr']
        price_local = original_price_inr * country['rate_from_inr']
        st.info(f'Estimated reference price: **{country["symbol"]}{price_local:,.0f} {country["currency"]}**')
    elif price_mode == 'Use estimated market price':
        st.warning('This model is not in the catalog. Choose a custom price to continue.')
        original_price_inr = None
    else:
        default_custom_price = round(
            model_entry['reference_price_inr'] * country['rate_from_inr']
            if model_entry
            else 40000 * country['rate_from_inr']
        )
        custom_price = st.number_input(
            f'Custom price ({country["currency"]})',
            min_value=1,
            value=max(default_custom_price, 1),
            step=max(round(500 * country['rate_from_inr']), 1),
            key=f'custom_price_{country["code"]}_{selected_brand}_{selected_model}',
        )
        original_price_inr = round(custom_price / country['rate_from_inr'])

    st.caption('Reference prices are curated estimates converted from INR, not live retailer listings.')

with condition_col:
    st.subheader('3. Age and condition')
    months_used = st.slider('Months used', 1, 47, 12)
    battery_health_pct = st.slider('Battery health', 65, 100, 85, format='%d%%')
    screen_condition = st.slider('Screen condition (1 = worn, 5 = like new)', 1, 5, 4)
    body_condition = st.slider('Body condition (1 = worn, 5 = like new)', 1, 5, 4)
    has_5g = st.checkbox('5G enabled', value=True)
    under_warranty = st.checkbox('Still under warranty')
    has_box_bill = st.checkbox('Box and bill available', value=True)

    st.markdown('')
    calculate = st.button('Calculate my estimate  →', type='primary', width='stretch')
    st.caption('For guidance only. Estimates are based on synthetic training data and curated reference prices.')

if calculate:
    if not brand.strip() or not model_name.strip():
        st.error('Enter both the brand and model name to continue.')
    elif original_price_inr is None:
        st.error('Automatic reference pricing is unavailable for this model. Enter a custom price.')
    else:
        user_input = pd.DataFrame([{
            'brand': brand,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'has_5g': int(has_5g),
            'original_price_inr': original_price_inr,
            'months_used': months_used,
            'screen_condition': screen_condition,
            'body_condition': body_condition,
            'battery_health_pct': battery_health_pct,
            'under_warranty': int(under_warranty),
            'has_box_bill': int(has_box_bill),
        }])
        predicted_inr = float(model.predict(user_input)[0])
        predicted_local = predicted_inr * country['rate_from_inr']
        st.success(f'### Estimated resale value: {country["symbol"]}{predicted_local:,.0f} {country["currency"]}')
        st.write(
            f'Likely range: {country["symbol"]}{predicted_local * 0.9:,.0f} – '
            f'{country["symbol"]}{predicted_local * 1.1:,.0f} {country["currency"]}'
        )
        st.caption(f'Estimate for {brand} {model_name}; it is not a live marketplace quote or guaranteed offer.')

with st.expander('View sample training data'):
    st.dataframe(load_data().head(10), width='stretch')
