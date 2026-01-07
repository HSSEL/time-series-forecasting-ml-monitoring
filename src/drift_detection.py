import pandas as pd
import numpy as np
from scipy.stats import ks_2samp



def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index
    expected: distribution de référence (historique)
    actual: distribution récente (prédictions)
    """

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Créer les bins à partir de la distribution attendue
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Éviter division par zéro
    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return psi


def ks_test(expected, actual):
    """Kolmogorov-Smirnov Test pour comparer distributions"""
    statistic, p_value = ks_2samp(expected, actual)
    return statistic, p_value

def detect_performance_drift(predictions_df, window=30):
    """
    Analyse performance drift via RMSE glissant
    predictions_df : DataFrame avec colonnes ['prediction_date', 'predicted_revenue', 'actual_revenue']
    """
    predictions_df = predictions_df.sort_values('prediction_date').reset_index(drop=True)
    
    rmse_list = []
    for i in range(len(predictions_df)-window+1):
        window_df = predictions_df.iloc[i:i+window]
        rmse = np.sqrt(np.mean((window_df['predicted_revenue'] - window_df['actual_revenue'])**2))
        rmse_list.append({'date': window_df['prediction_date'].iloc[-1], 'rmse': rmse})
    
    return pd.DataFrame(rmse_list)
