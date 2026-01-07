"""
Script pour générer des prédictions avec le modèle LSTM
Fichier: generate_predictions.py
"""

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configurations
MODEL_PATH = 'models/lstm_model.pkl'
DATA_PATH = 'data/daily_data_preprocessed.csv'
PREDICTIONS_PATH = 'predictions.csv'
LOOKBACK = 14  # nombre de jours historiques pour les lags

# Variables globales
model = None
scaler = None
historical_data = None
feature_columns = None

# ----------------------- Fonctions -----------------------

def load_model_and_data():
    """Charge le modèle et les données historiques"""
    global model, scaler, historical_data, feature_columns
    try:
        # Charger le modèle
        logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data.get('scaler', MinMaxScaler())
            feature_columns = model_data.get('feature_columns', None)
        logger.info("Modèle chargé avec succès")

        # Charger les données historiques
        historical_data = pd.read_csv(DATA_PATH)
        historical_data.rename(columns={'InvoiceDate':'date','Revenue':'revenue'}, inplace=True)
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.sort_values('date').reset_index(drop=True)

        # Si feature_columns n'est pas défini
        if feature_columns is None:
            feature_columns = [col for col in historical_data.columns if col not in ['date','revenue','day_of_week']]

        logger.info(f"Colonnes attendues par le modèle: {feature_columns}")
        logger.info(f"Données chargées: {len(historical_data)} enregistrements")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        return False

def calculate_lag_features(df, target_col='revenue', lags=[1,7,14]):
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def calculate_rolling_features(df, target_col='revenue', windows=[7]):
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_{window}'] = df[target_col].rolling(window=window).mean()
    return df

def add_calendar_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.day_name()
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for day in days:
        df[f'day_of_week_{day}'] = (df['day_of_week'] == day).astype(int)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df

def prepare_features_for_date(target_date, data):
    """Prépare les features pour la date cible"""
    df = data.copy()
    new_row = pd.DataFrame({'date':[target_date]})
    df = pd.concat([df,new_row], ignore_index=True).sort_values('date').reset_index(drop=True)

    # Lag, rolling, calendar
    df = calculate_lag_features(df, 'revenue', [1,7,14])
    df = calculate_rolling_features(df, 'revenue', [7])
    df = add_calendar_features(df)

    # Ajouter colonnes manquantes
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Extraire la ligne cible et remplacer NaN par 0
    target_row = df[df['date']==target_date].copy()
    target_row[feature_columns] = target_row[feature_columns].fillna(0)

    return target_row

def create_lstm_input(features_df):
    """Prépare l'input LSTM (1,1,n_features)"""
    features = features_df[feature_columns].values
    features_scaled = scaler.transform(features)
    return features_scaled.reshape(1,1,-1)

def make_prediction(target_date):
    features_df = prepare_features_for_date(target_date, historical_data)
    lstm_input = create_lstm_input(features_df)
    prediction = model.predict(lstm_input, verbose=0)
    return float(prediction[0][0])

def save_prediction(date, predicted_revenue):
    prediction_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction_date': date.strftime('%Y-%m-%d'),
        'predicted_revenue': predicted_revenue
    }
    if os.path.exists(PREDICTIONS_PATH):
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
        predictions_df = pd.concat([predictions_df, pd.DataFrame([prediction_entry])], ignore_index=True)
    else:
        predictions_df = pd.DataFrame([prediction_entry])
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    logger.info(f"Prédiction sauvegardée pour {date.strftime('%Y-%m-%d')}")

# ----------------------- Main -----------------------

if __name__ == "__main__":
    if not load_model_and_data():
        logger.error("Impossible de charger le modèle ou les données")
        sys.exit(1)

    # Exemple : prédire le lendemain du dernier jour connu
    last_date = historical_data['date'].max()
    next_date = last_date + pd.Timedelta(days=1)
    predicted_revenue = make_prediction(next_date)
    logger.info(f"Predicted revenue pour {next_date.strftime('%Y-%m-%d')}: {predicted_revenue:.2f}")
    save_prediction(next_date, predicted_revenue)
