import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from datetime import datetime

# Ajouter le chemin parent au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.drift_detection import calculate_psi, ks_test, detect_performance_drift

# --- Titre du Dashboard ---
st.title(" Monitoring & Drift Dashboard")
st.markdown("""
Ce dashboard permet de suivre l'évolution des prédictions, détecter le **data drift** avec le PSI
et surveiller la performance du modèle avec les vraies valeurs.
""")

# --- Chargement des données ---
try:
    historical = pd.read_csv('data/daily_data_preprocessed.csv')
    historical.rename(columns={'InvoiceDate': 'date'}, inplace=True)
    historical['date'] = pd.to_datetime(historical['date'])
except FileNotFoundError:
    st.error("Fichier historique non trouvé : data/daily_data_preprocessed.csv")
    st.stop()

try:
    predictions = pd.read_csv('predictions.csv')
    predictions.rename(columns={'prediction_date': 'date'}, inplace=True)
    predictions['date'] = pd.to_datetime(predictions['date'])
except FileNotFoundError:
    st.error("Fichier de prédictions non trouvé : predictions.csv")
    st.stop()

# --- Calcul du PSI et création des logs ---
psi_history = []
alerts = []

for date, group in predictions.groupby('date'):
    daily_preds = group['predicted_revenue'].dropna()
    if len(daily_preds) >= 3:
        psi = calculate_psi(historical['Revenue'], daily_preds)
        psi_history.append({'date': date, 'psi': psi})
        if psi > 0.3:
            alerts.append({'date': date, 'psi': psi, 'timestamp': datetime.now()})
            
# Sauvegarde des logs
if psi_history:
    psi_df = pd.DataFrame(psi_history).sort_values('date')
    psi_df.to_csv('psi_log.csv', index=False)

if alerts:
    alerts_df = pd.DataFrame(alerts)
    alerts_file = 'alerts_log.csv'
    if os.path.exists(alerts_file):
        alerts_df.to_csv(alerts_file, mode='a', index=False, header=False)
    else:
        alerts_df.to_csv(alerts_file, index=False)
    st.warning(f"{len(alerts)} alertes détectées (PSI > 0.3) !")

# --- Section Data Drift ---
st.subheader("📈 Évolution du PSI dans le temps")

if psi_history:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(psi_df['date'], psi_df['psi'], marker='o', label='PSI')
    # Seuils
    ax.axhline(0.1, linestyle='--', linewidth=1, label='PSI = 0.1 (Stable)')
    ax.axhline(0.2, linestyle='--', linewidth=1, label='PSI = 0.2 (Drift léger)')
    ax.axhline(0.3, linestyle='--', linewidth=1, label='PSI = 0.3 (Drift critique)')
    ax.set_xlabel("Date")
    ax.set_ylabel("PSI")
    ax.set_title("Évolution du PSI")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Pas assez de données pour tracer l’évolution du PSI.")

# --- Section Performance Drift ---
st.subheader("Suivi de performance ")
if 'actual_revenue' in predictions.columns and predictions['actual_revenue'].notna().any():
    perf_df = detect_performance_drift(predictions)
    st.line_chart(perf_df.set_index('date')['rmse'])
else:
    st.info("Les valeurs réelles (actual_revenue) sont nécessaires pour suivre la performance.")

# --- Section Prédictions vs Historique ---
st.subheader("📉 Prédictions vs Historique")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(historical['date'], historical['Revenue'], label='Historique', color='blue')
ax.plot(predictions['date'], predictions['predicted_revenue'], label='Prédictions', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue')
ax.set_title('Comparaison Prédictions vs Historique')
ax.legend()
st.pyplot(fig)

# --- Bouton Simulation Retraining ---
st.subheader("⚙️ Retraining du modèle (simulation)")
if st.button("Lancer retraining"):
    st.info("Simulation de retraining déclenchée !")
    # Ici, tu pourrais appeler un script python pour ré-entraîner le modèle et régénérer les prédictions
    # Exemple: os.system("python generate_predictions.py")
    st.success("Retraining terminé (simulation). Nouvelles prédictions disponibles.")
