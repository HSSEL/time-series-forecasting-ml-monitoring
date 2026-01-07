"""
API Flask pour prédiction de revenue avec modèle LSTM
Fichier: api/app.py
"""

from flask import Flask, request, jsonify
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
    handlers=[
        logging.FileHandler('api/api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialisation Flask
app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/lstm_model.pkl'
DATA_PATH = 'data/daily_data_preprocessed.csv'
PREDICTIONS_PATH = 'predictions.csv'
LOOKBACK = 14  # Nombre de jours historiques nécessaires


# Variables globales
model = None
scaler = None
historical_data = None
feature_columns = None


def load_model_and_data():
    """Charge le modèle LSTM et les données historiques au démarrage"""
    global model, scaler, historical_data, feature_columns
    
    try:
        # Charger le modèle LSTM
        logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data.get('scaler', MinMaxScaler())
            feature_columns = model_data.get('feature_columns', None)
        
        # Si feature_columns est None, utiliser toutes les colonnes sauf date et revenue
        if feature_columns is None:
            feature_columns = [col for col in historical_data.columns 
                               if col not in ['date', 'revenue', 'day_of_week']]
        
        logger.info(f"Colonnes attendues par le modèle: {feature_columns}")
        logger.info("Modèle chargé avec succès")
        
        # Charger les données historiques
        logger.info(f"Chargement des données depuis {DATA_PATH}")
        historical_data = pd.read_csv(DATA_PATH)
        historical_data.rename(columns={
               'InvoiceDate': 'date',
               'Revenue': 'revenue'
        }, inplace=True)

        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Données chargées: {len(historical_data)} enregistrements")
        logger.info(f"Période: {historical_data['date'].min()} à {historical_data['date'].max()}")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé: {e}")
        return False
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        return False


def calculate_lag_features(data, target_col='revenue', lags=[1, 7, 14]):
    """Calcule les features de lag pour les données"""
    df = data.copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def calculate_rolling_features(data, target_col='revenue', windows=[7, 14]):
    """Calcule les features rolling (moyenne mobile)"""
    df = data.copy()
    
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
    
    return df

def add_calendar_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.day_name()  # ex: Monday, Tuesday...
    
    # One-hot encoding
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        df[f'day_of_week_{day}'] = (df['day_of_week'] == day).astype(int)
    
    # Jour du mois
    df['day_of_month'] = df['date'].dt.day
    
    # Mois
    df['month'] = df['date'].dt.month
    
    return df




def prepare_features_for_date(target_date, data):
    df = data.copy()
    new_row = pd.DataFrame({'date':[target_date]})
    df = pd.concat([df,new_row], ignore_index=True).sort_values('date').reset_index(drop=True)
    
    # Calculer lags et rolling
    df = calculate_lag_features(df, 'revenue', [1,7,14])
    df = calculate_rolling_features(df, 'revenue', [7])
    df = add_calendar_features(df)
    
    # Ajouter colonnes manquantes
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Remplacer NaN par 0 pour la nouvelle ligne
    target_row = df[df['date']==target_date].copy()
    target_row[feature_columns] = target_row[feature_columns].fillna(0)
    
    return target_row

def create_lstm_input(features_df):
    features = features_df[feature_columns].values
    features_scaled = scaler.transform(features)  # Assurez-vous que scaler est bien chargé avec ces colonnes
    return features_scaled.reshape(1, 1, -1)


def make_prediction(target_date):
    """
    Fait une prédiction pour une date donnée
    
    Args:
        target_date: datetime - date cible
    
    Returns:
        float - revenue prédit
    """
    # Préparer les features
    features_df = prepare_features_for_date(target_date, historical_data)
    
    # Créer l'input LSTM
    lstm_input = create_lstm_input(features_df)
    
    # Faire la prédiction
    prediction = model.predict(lstm_input, verbose=0)
    predicted_revenue = float(prediction[0][0])
    
    # Dénormaliser si nécessaire
    # Si le scaler a été appliqué au target, il faut inverse_transform
    # Pour cet exemple, on suppose que la prédiction est déjà à la bonne échelle
    
    return predicted_revenue


def save_prediction(date, predicted_revenue):
    """Sauvegarde la prédiction dans un fichier CSV"""
    try:
        prediction_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction_date': date.strftime('%Y-%m-%d'),
            'predicted_revenue': predicted_revenue
        }
        
        # Charger les prédictions existantes ou créer un nouveau DataFrame
        if os.path.exists(PREDICTIONS_PATH):
            predictions_df = pd.read_csv(PREDICTIONS_PATH)
            predictions_df = pd.concat([predictions_df, pd.DataFrame([prediction_entry])], ignore_index=True)
        else:
            predictions_df = pd.DataFrame([prediction_entry])
        
        # Sauvegarder
        predictions_df.to_csv(PREDICTIONS_PATH, index=False)
        logger.info(f"Prédiction sauvegardée dans {PREDICTIONS_PATH}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la prédiction: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé de l'API"""
    status = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'data_loaded': historical_data is not None
    }
    
    if historical_data is not None:
        status['data_range'] = {
            'start': historical_data['date'].min().strftime('%Y-%m-%d'),
            'end': historical_data['date'].max().strftime('%Y-%m-%d'),
            'records': len(historical_data)
        }
    
    logger.info("Health check appelé")
    return jsonify(status), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    
    Input JSON:
        {"date": "YYYY-MM-DD"}
    ou
        {"dates": ["YYYY-MM-DD", "YYYY-MM-DD", ...]}
    
    Output JSON:
        {"date": "YYYY-MM-DD", "predicted_revenue": float}
    ou
        {"predictions": [{"date": "...", "predicted_revenue": ...}, ...]}
    """
    try:
        # Vérifier que le modèle est chargé
        if model is None or historical_data is None:
            logger.error("Modèle ou données non chargés")
            return jsonify({
                'error': 'Modèle non initialisé',
                'message': 'Le service n\'est pas prêt'
            }), 503
        
        # Récupérer les données de la requête
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Données manquantes',
                'message': 'Le corps de la requête doit contenir du JSON'
            }), 400
        
        # Cas 1: Prédiction pour une seule date
        if 'date' in data:
            date_str = data['date']
            
            try:
                target_date = pd.to_datetime(date_str)
            except Exception as e:
                return jsonify({
                    'error': 'Format de date invalide',
                    'message': f'La date doit être au format YYYY-MM-DD. Erreur: {str(e)}'
                }), 400
            
            logger.info(f"Prédiction demandée pour {target_date.strftime('%Y-%m-%d')}")
            
            # Faire la prédiction
            predicted_revenue = make_prediction(target_date)
            
            # Sauvegarder la prédiction
            save_prediction(target_date, predicted_revenue)
            
            response = {
                'date': target_date.strftime('%Y-%m-%d'),
                'predicted_revenue': round(predicted_revenue, 2)
            }
            
            logger.info(f"Prédiction: {predicted_revenue:.2f}")
            return jsonify(response), 200
        
        # Cas 2: Prédictions pour plusieurs dates
        elif 'dates' in data:
            dates_str = data['dates']
            
            if not isinstance(dates_str, list):
                return jsonify({
                    'error': 'Format invalide',
                    'message': 'Le champ "dates" doit être une liste'
                }), 400
            
            predictions = []
            
            for date_str in dates_str:
                try:
                    target_date = pd.to_datetime(date_str)
                    predicted_revenue = make_prediction(target_date)
                    save_prediction(target_date, predicted_revenue)
                    
                    predictions.append({
                        'date': target_date.strftime('%Y-%m-%d'),
                        'predicted_revenue': round(predicted_revenue, 2)
                    })
                    
                except Exception as e:
                    logger.error(f"Erreur pour la date {date_str}: {e}")
                    predictions.append({
                        'date': date_str,
                        'error': str(e)
                    })
            
            logger.info(f"Prédictions pour {len(predictions)} dates")
            return jsonify({'predictions': predictions}), 200
        
        else:
            return jsonify({
                'error': 'Paramètres manquants',
                'message': 'La requête doit contenir "date" ou "dates"'
            }), 400
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
        return jsonify({
            'error': 'Erreur interne',
            'message': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """Endpoint pour obtenir des informations sur le modèle et les données"""
    try:
        info_data = {
            'model': {
                'path': MODEL_PATH,
                'loaded': model is not None
            },
            'data': {
                'path': DATA_PATH,
                'loaded': historical_data is not None
            }
        }
        
        if historical_data is not None:
            info_data['data']['records'] = len(historical_data)
            info_data['data']['date_range'] = {
                'start': historical_data['date'].min().strftime('%Y-%m-%d'),
                'end': historical_data['date'].max().strftime('%Y-%m-%d')
            }
            info_data['data']['columns'] = list(historical_data.columns)
        
        if model is not None:
            info_data['model']['architecture'] = str(type(model))
        
        return jsonify(info_data), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Gestion des erreurs 404"""
    return jsonify({
        'error': 'Endpoint non trouvé',
        'message': 'Cet endpoint n\'existe pas'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs 500"""
    logger.error(f"Erreur 500: {error}")
    return jsonify({
        'error': 'Erreur interne du serveur',
        'message': 'Une erreur s\'est produite'
    }), 500


if __name__ == '__main__':
    # Charger le modèle et les données au démarrage
    logger.info("Démarrage de l'API Flask LSTM")
    
    if not load_model_and_data():
        logger.error("Impossible de charger le modèle ou les données")
        sys.exit(1)
    
    logger.info("API prête à recevoir des requêtes")
    
    # Lancer l'application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Mettre à False en production
    )