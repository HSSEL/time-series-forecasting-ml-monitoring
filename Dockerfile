# Base image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Lancer Streamlit au démarrage du container
CMD ["streamlit", "run", "dashboard/monitoring.py", "--server.port=8501", "--server.address=0.0.0.0"]
