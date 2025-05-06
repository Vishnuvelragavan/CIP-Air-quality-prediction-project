import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def get_season(month):
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else: return 3

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(get_season)

    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    for p in pollutants:
        df[p] = df[p].fillna(df[p].median())

    df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6)
    df['NOx_ratio'] = (df['NO'] + df['NO2']) / (df['NOx'] + 1e-6)
    df['PM2.5_NO2'] = df['PM2.5'] * df['NO2']
    df['SO2_CO_ratio'] = df['SO2'] / (df['CO'] + 1e-6)
    df['O3_PM10'] = df['O3'] * df['PM10']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['AQI'], inplace=True)
    return df

def prepare_data(df):
    X = df.drop(['AQI', 'AQI_Bucket', 'City', 'Date'], axis=1, errors='ignore')
    y = df['AQI'].values.reshape(-1, 1)

    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y).ravel()

    X_train, _, y_train, _ = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled, y_train, scaler, target_scaler, X.columns

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def main():
    st.set_page_config(page_title="AQI Predictor", layout="wide")
    st.markdown("<h1 style='text-align:center; color:#0073e6;'>🌫 AQI Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#4CAF50;'>Predict Air Quality Index using pollutant levels</h4>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("📂 Load Dataset & Train Model"):
        data_path = r"C:\Users\S.GHIRIDHARAN\Downloads\city_day.csv"  # Update path if needed
        df = load_data(data_path)
        X_train_scaled, y_train, scaler, target_scaler, feature_cols = prepare_data(df)
        model = train_model(X_train_scaled, y_train)
        st.success("✅ Model trained successfully!")

    left, center, right = st.columns([1, 2, 1])

    with left:
        st.markdown("### 🧠 AQI Info")
        st.info("""
        - AQI < 50 → Good
        - 51–100 → Satisfactory
        - 101–200 → Moderate
        - 201–300 → Poor
        - 301–400 → Very Poor
        - >400 → Severe
        """)
        st.markdown("### 🧼 Tips to Reduce AQI")
        st.success("""
        - Use air purifiers
        - Avoid burning waste
        - Use masks
        - Drive less
        """)

    with right:
        st.markdown("### 🌍 Environmental Impact")
        st.warning("""
        High AQI:
        - Respiratory issues
        - Low visibility
        - Harmful to plants
        """)
        st.markdown("### 📌 Fun Fact")
        st.info("PM2.5 particles are 30x smaller than a human hair!")

    with center:
        st.subheader("🔢 Enter Pollutant Levels")

        user_inputs = {}
        pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        default_values = ['60', '100', '20', '30', '40', '10', '1.0', '10', '100', '1', '2', '1']

        for i, pollutant in enumerate(pollutants):
            user_inputs[pollutant] = st.text_input(f"{pollutant} (µg/m³)", value=default_values[i])

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])

        with c2:
            if st.button("📊 Predict AQI"):
                try:
                    inputs = {k: float(v) for k, v in user_inputs.items()}

                    # Feature engineering
                    inputs['PM_ratio'] = inputs['PM2.5'] / (inputs['PM10'] + 1e-6)
                    inputs['NOx_ratio'] = (inputs['NO'] + inputs['NO2']) / (inputs['NOx'] + 1e-6)
                    inputs['PM2.5_NO2'] = inputs['PM2.5'] * inputs['NO2']
                    inputs['SO2_CO_ratio'] = inputs['SO2'] / (inputs['CO'] + 1e-6)
                    inputs['O3_PM10'] = inputs['O3'] * inputs['PM10']
                    inputs['Season'] = 1  # default season
                    inputs['Month'] = 4   # default month

                    input_df = pd.DataFrame([inputs])
                    for col in feature_cols:
                        if col not in input_df.columns:
                            input_df[col] = 0

                    input_scaled = scaler.transform(input_df[feature_cols])
                    pred_norm = model.predict(input_scaled)[0]
                    pred_aqi = int(target_scaler.inverse_transform([[pred_norm]])[0][0])
                    category = get_aqi_category(pred_aqi)

                    st.markdown(f"""
                    <div style='background-color:#eaffea; padding:20px; border-radius:10px; text-align:center;'>
                        <h2 style='color:#2e7d32;'>🌟 Predicted AQI: {pred_aqi}</h2>
                        <h3 style='color:#1565c0;'>Category: {category}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                except ValueError:
                    st.error("Please enter valid numeric values only.")

        with c2:
            if st.button("🧹 Clear All Fields"):
                st.experimental_rerun()

    st.markdown("---")
    st.caption("Developed with ❤ using Streamlit")

if __name__ == "_main_":
    main()
