import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# -----------------------------
# Season Encoding
# -----------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall


# -----------------------------
# Load and Preprocess Data
# -----------------------------
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df['City'] == 'Hyderabad'].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Season'] = df['Month'].apply(get_season)

    # Fill missing pollutant values
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    for p in pollutants:
        df[p] = df[p].fillna(df[p].median())

    # Derived Features
    df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6)
    df['NOx_ratio'] = (df['NO'] + df['NO2']) / (df['NOx'] + 1e-6)
    df['PM2.5_NO2'] = df['PM2.5'] * df['NO2']
    df['SO2_CO_ratio'] = df['SO2'] / (df['CO'] + 1e-6)
    df['O3_PM10'] = df['O3'] * df['PM10']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['AQI'], inplace=True)
    return df


# -----------------------------
# Feature Engineering (Lag + Rolling)
# -----------------------------
def add_advanced_features(df):
    df = df.sort_values('Date')
    features = ['AQI', 'PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'O3']

    for feat in features:
        for lag in [1, 2, 3]:
            df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
        df[f'{feat}_roll7_mean'] = df[feat].rolling(window=7).mean()
        df[f'{feat}_roll7_std'] = df[feat].rolling(window=7).std()

    df.dropna(inplace=True)
    return df


# -----------------------------
# Prepare Features and Normalized Target
# -----------------------------
def prepare_data(df):
    X = df.drop(['AQI', 'AQI_Bucket', 'City', 'Date'], axis=1, errors='ignore')
    y = df['AQI'].values.reshape(-1, 1)

    # Normalize target (Min-Max)
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, target_scaler


# -----------------------------
# Train Random Forest
# -----------------------------
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n📊 Evaluation on Normalized AQI:")
    print(f"R² Score : {r2:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}")


# -----------------------------
# Main Execution
# -----------------------------
def main():
    path = r"C:\Users\S.GHIRIDHARAN\Downloads\city_day.csv"  # Replace with your actual path
    df = load_data(path)

    # Print dataset size and sample values
    print(f"📦 Dataset size: {df.shape}")
    print("🔍 Sample data:\n", df.head())

    df = add_advanced_features(df)

    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # -----------------------------
    # Plot Actual vs Predicted Scatter
    # -----------------------------
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_test)), y_test, color='slateblue', alpha=0.6, label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='lightcoral', alpha=0.6, label='Predicted')
    plt.xlabel("Observations")
    plt.ylabel("Targets")
    plt.title("Actual and predicted values")
    plt.legend()
    plt.tight_layout()
    plt.show()


if _name_ == "_main_":
    main()
