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

DATA_FILE = Path(__file__).resolve().parent / 'used_phones.csv'
MODEL_FILE = Path(__file__).resolve().parent / 'used_phone_price_model.joblib'


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


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

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': rmse,
        'R2': r2_score(y_test, y_pred),
    }

    return model, metrics


def save_model(model):
    joblib.dump(model, MODEL_FILE)


def load_or_train_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)

    model, _ = build_model_and_metrics()
    save_model(model)
    return model


st.set_page_config(
    page_title='Used Mobile Price Predictor',
    page_icon='📱',
    layout='wide',
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None,
    },
)

st.markdown(
    """
    <style>
    div[data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    div[data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('Used Mobile Resale Price Predictor')
st.caption('Estimate resale value for a second-hand phone from key device characteristics.')

if not DATA_FILE.exists():
    st.error(f'Data file not found: {DATA_FILE}')
    st.stop()

model, model_metrics = build_model_and_metrics()

with st.sidebar:
    st.header('Phone details')
    brand_options = sorted(load_data()['brand'].unique().tolist())
    brand = st.selectbox('Brand', brand_options)
    original_price_inr = st.number_input('Original purchase price (₹)', min_value=5000, max_value=200000, value=40000, step=500)
    ram_gb = st.slider('RAM (GB)', 4, 16, 8, step=2)
    storage_gb = st.selectbox('Storage', [64, 128, 256, 512])
    has_5g = st.radio('5G enabled?', [0, 1], index=0, format_func=lambda x: 'Yes' if x == 1 else 'No')
    months_used = st.slider('Months used', 1, 47, 12)
    screen_condition = st.slider('Screen condition (1-5)', 1, 5, 4)
    body_condition = st.slider('Body condition (1-5)', 1, 5, 4)
    battery_health_pct = st.slider('Battery health (%)', 65, 100, 85)
    under_warranty = st.radio('Under warranty?', [0, 1], index=0, format_func=lambda x: 'Yes' if x == 1 else 'No')
    has_box_bill = st.radio('Has box/bill?', [0, 1], index=1, format_func=lambda x: 'Yes' if x == 1 else 'No')

    st.markdown('---')
    st.subheader('Model quality')
    st.metric('MAE', f'₹{model_metrics["MAE"]:,.0f}')
    st.metric('RMSE', f'₹{model_metrics["RMSE"]:,.0f}')
    st.metric('R²', f'{model_metrics["R2"]:.3f}')

with st.form('prediction_form'):
    submitted = st.form_submit_button('Predict resale price', width='stretch')

if submitted:
    user_input = pd.DataFrame([
        {
            'brand': brand,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'has_5g': has_5g,
            'original_price_inr': original_price_inr,
            'months_used': months_used,
            'screen_condition': screen_condition,
            'body_condition': body_condition,
            'battery_health_pct': battery_health_pct,
            'under_warranty': under_warranty,
            'has_box_bill': has_box_bill,
        }
    ])

    prediction = model.predict(user_input)[0]
    price_range = prediction * 0.9, prediction * 1.1

    st.subheader('Estimated resale price')
    st.metric('Predicted value', f'₹{prediction:,.0f}')
    st.write(f'Likely range: ₹{price_range[0]:,.0f} to ₹{price_range[1]:,.0f}')

    st.info('This estimate is generated using a machine learning model trained on synthetic used-phone resale data.')

st.subheader('Dataset snapshot')
raw_df = load_data().head(10)
st.dataframe(raw_df, width='stretch')
