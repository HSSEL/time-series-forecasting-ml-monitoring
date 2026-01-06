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
* **Période couverte :** 2009 – 2011
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

## 📈 Résultats (exemple)

* Le modèle **XGBoost** offre le meilleur compromis biais/variance sur les données agrégées.
* Le modèle **LSTM** capture efficacement les dépendances temporelles longues.
* Amélioration significative de l’erreur de prévision par rapport au modèle de référence statistique.

*(Les résultats chiffrés détaillés sont présentés dans les notebooks et le dashboard.)*

---

## 🧪 Monitoring & Détection de dérive

Un système de monitoring est mis en place afin de :

* Détecter la **dérive des données** (Data Drift) :

  * Population Stability Index (PSI)
  * Test de Kolmogorov-Smirnov (KS)
* Surveiller la **performance du modèle** dans le temps
* Simuler des scénarios de **retraining automatique**

Un dashboard interactif permet de visualiser l’évolution des distributions et des métriques.

---

## 🚀 Déploiement

* **API :** Flask
* **Endpoints :**

  * `/predict` – prévision des volumes futurs
  * `/health` – vérification de l’état du service
* **Containerisation :** Docker
* **Versioning :** modèles sauvegardés et historisés

---

## 🗂 Structure du projet

```
📦 time-series-forecasting-monitoring
 ┣ 📂 data
 ┣ 📂 notebooks
 ┣ 📂 src
 ┃ ┣ preprocessing.py
 ┃ ┣ forecasting.py
 ┃ ┣ drift_detection.py
 ┃ ┗ retraining.py
 ┣ 📂 api
 ┃ ┗ app.py
 ┣ 📂 dashboard
 ┃ ┗ monitoring.py
 ┣ 📄 README.md
 ┣ 📄 requirements.txt
 ┣ 📄 Dockerfile
 ┗ 📄 architecture.png
```

---

## 🛠 Technologies utilisées

* **Langage :** Python
* **Data Science :** Pandas, NumPy, Scikit-learn
* **Time Series :** Statsmodels, Prophet
* **Deep Learning :** TensorFlow / Keras
* **ML Ops :** MLflow, Evidently AI
* **Big Data (optionnel) :** PySpark
* **API :** Flask
* **Dashboard :** Streamlit
* **Conteneurisation :** Docker

---

## 🔮 Améliorations futures

* Intégration de données exogènes (promotions, jours fériés)
* Automatisation complète du retraining
* Déploiement cloud (AWS / Azure)
* Passage à un pipeline temps réel

---

## 👩‍💻 Auteur

**Hafssa El Mouddane**
Ingénieure Data Science & Big Data
📎 GitHub : HSSEL
📎 LinkedIn : linkedin.com/in/hafssa-el-mouddane-815ba7251



