# time-series-forecasting-ml-monitoring
# 📊 Système de prévision et de monitoring intelligent des comportements clients

*(Time Series Forecasting & ML Monitoring System)*

## 🧠 Contexte & Problématique métier

Dans de nombreux secteurs (banque, retail, télécommunications), l’anticipation du comportement client est un enjeu stratégique majeur.
La capacité à **prévoir l’évolution des volumes d’activité** permet d’optimiser la gestion commerciale, d’anticiper les pics de demande et de détecter rapidement les changements anormaux de comportement.

**Problématique :**

> *Comment prévoir l’évolution du volume de transactions clients dans le temps tout en garantissant la fiabilité et la robustesse du modèle face à la dérive des données en production ?*

---

## 🎯 Objectifs du projet

Ce projet vise à concevoir un **système data de bout en bout** capable de :

* Prévoir les volumes transactionnels clients via des modèles de séries temporelles
* Comparer approches statistiques, Machine Learning et Deep Learning
* Déployer un service de prédiction via une API
* Mettre en place un système de **monitoring et de détection de dérive**
* Aider à la prise de décision métier à partir des prévisions

---

## 📂 Données utilisées

* **Dataset :** Online Retail II
* **Source :** Kaggle
* **Nature des données :** Transactions clients horodatées
* **Période couverte :** Décembre 2009 – Décembre 2010
* **Variables clés :**

  * `InvoiceDate`
  * `CustomerID`
  * `Quantity`
  * `UnitPrice`
  * `Country`

Les données nécessitent un nettoyage approfondi (valeurs négatives, retours produits, données manquantes), reproduisant un contexte réel en entreprise.

---

## 🧩 Méthodologie

Le projet suit la méthodologie **CRISP-DM** :

1. **Business Understanding**
   Définition du cas d’usage, des KPI métier et des objectifs de prévision.

2. **Data Understanding**
   Analyse exploratoire, détection de tendance, saisonnalité et anomalies.

3. **Data Preparation**
   Nettoyage, agrégation temporelle, feature engineering (lags, rolling statistics, variables calendaires).

4. **Modeling**
   Implémentation et comparaison de plusieurs modèles :

   * SARIMA (statistique)
   * XGBoost Regressor (Machine Learning)
   * LSTM (Deep Learning)

5. **Evaluation**
   Backtesting temporel et évaluation via :

   * RMSE
   * MAE
   * MAPE

6. **Deployment**
   Mise à disposition des prédictions via une API Flask.

7. **Monitoring & Drift Detection**
   Surveillance des données et des performances du modèle en production.

---

## 🤖 Modèles implémentés

| Type             | Modèle            |
| ---------------- | ----------------- |
| Statistique      | SARIMA            |
| Machine Learning | XGBoost Regressor |
| Deep Learning    | LSTM              |

Les modèles sont comparés sur la base de métriques quantitatives et de leur capacité à généraliser dans le temps.

---

## 📈 Résultats

Les expérimentations montrent que les modèles Machine Learning et Deep Learning 
(LSTM, XGBoost) offrent de meilleures performances que les approches statistiques 
classiques sur les données agrégées.


---

## 🧪 Monitoring & Détection de dérive

Le dashboard interactif (Streamlit) permet de :
* Visualiser l’évolution des distributions des données et des métriques du modèle
* Détecter la dérive des données (Data Drift) :
  * Population Stability Index (PSI)
  * Test de Kolmogorov-Smirnov (KS)
* Surveiller la performance des modèles dans le temps
* Simuler des scénarios de retraining automatique


---

## 🚀 Déploiement & Accès au Dashboard

* **API :** Flask
* **Endpoints :**
  * `/predict` – prévision des volumes futurs
  * `/health` – vérification de l’état du service
* **Dashboard interactif :** Streamlit
  * Accessible localement via : http://localhost:8501
* **Containerisation :** Docker
* **Versioning :** modèles sauvegardés et historisés


---
## 🐳 Lancer le projet avec Docker

1. **Construire l’image Docker :**
```bash
docker build -t ts-forecasting-dashboard .

2. **Lancer le container :**
docker run -p 8501:8501 ts-forecasting-dashboard

3.Accéder au dashboard :
Ouvrir dans votre navigateur :http://127.0.0.1:8501/


```markdown
## 📂 Notebooks

* `notebooks/01_preprocessing.ipynb` – Analyse et nettoyage des données
* `notebooks/02_modeling.ipynb` – Implémentation et comparaison des modèles
* `notebooks/03_monitoring.ipynb` – Simulation de dérive et visualisation

Le système de monitoring inclut :

- Calcul quotidien du PSI (Population Stability Index)
- Historisation automatique des valeurs PSI
- Déclenchement d’alertes si PSI > 0.3
- Visualisation de l’évolution du PSI dans le temps
- Simulation de retraining via dashboard Streamlit

## 🗂 Structure du projet

```
📦 time-series-forecasting-ml-monitoring
 ┣ 📂 api
 ┃ ┣ 📄 app.py
 ┃ ┗ 📄 api.log
 ┣ 📂 dashboard
 ┃ ┗ 📄 monitoring.py
 ┣ 📂 data
 ┃ ┗ 📄 daily_data_preprocessed.csv
 ┣ 📂 models
 ┃ ┗ 📄 lstm_model.pkl
 ┣ 📂 notebooks
 ┃ ┣ 📄 01_eda.ipynb
 ┃ ┣ 📄 02_preprocessing.ipynb
 ┃ ┗ 📄 03_modeling.ipynb
 ┣ 📂 src
 ┃ ┗ 📄 drift_detection.py
 ┣ 📄 predictions.csv
 ┣ 📄 psi_log.csv
 ┣ 📄 alerts_log.csv
 ┣ 📄 generate_predictions.py
 ┣ 📄 Dockerfile
 ┣ 📄 requirements.txt
 ┣ 📄 architecture.png
 ┗ 📄 README.md

```



## 🛠 Technologies utilisées

- Langage : Python
- Data Science : Pandas, NumPy, Scikit-learn
- Time Series : Statsmodels
- Deep Learning : TensorFlow / Keras
- API : Flask
- Monitoring & Dashboard : Streamlit
- Containerisation : Docker
- Visualisation : Matplotlib


---

## 🔮 Améliorations futures

* Intégration de données exogènes (promotions, jours fériés)
* Automatisation complète du retraining
* Déploiement cloud (AWS / Azure)
* Passage à un pipeline temps réel

---

## 👩‍💻 Auteur

**Hafssa El Mouddane**
Ingénieure Data Science 
📎 GitHub : HSSEL
📎 LinkedIn : linkedin.com/in/hafssa-el-mouddane-815ba7251



