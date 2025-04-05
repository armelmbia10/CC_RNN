import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPooling1D, Flatten
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(page_title="Prédiction des cours d'Apple", layout="wide")
st.title("Prédiction des cours d'Apple avec CNN1D et GRU")

# Fonction pour télécharger les données
@st.cache_data
def load_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=4*365)  # 4 ans de données
    data = yf.download('AAPL', start=start_date, end=end_date)
    return data

# Fonction pour préparer les données
def prepare_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:(i + lookback), 0])
        y.append(scaled_data[i + lookback, 0])
    
    return np.array(X), np.array(y), scaler

# Fonction pour créer le modèle CNN1D
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Fonction pour créer le modèle GRU
def create_gru_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        GRU(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Interface Streamlit
st.sidebar.header("Paramètres")
lookback = st.sidebar.slider("Période de lookback", 30, 90, 60)
train_size = st.sidebar.slider("Taille de l'ensemble d'entraînement (%)", 70, 90, 80)

# Chargement des données
data = load_data()
st.subheader("Données historiques d'Apple")
st.line_chart(data['Close'])

# Préparation des données
X, y, scaler = prepare_data(data, lookback)
train_size = int(len(X) * train_size / 100)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape des données pour les modèles
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Création et entraînement des modèles
cnn_model = create_cnn_model((lookback, 1))
gru_model = create_gru_model((lookback, 1))

st.subheader("Entraînement des modèles")
with st.spinner("Entraînement en cours..."):
    cnn_history = cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    gru_history = gru_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Prédictions
cnn_predictions = cnn_model.predict(X_test)
gru_predictions = gru_model.predict(X_test)

# Inversion de la normalisation
cnn_predictions = scaler.inverse_transform(cnn_predictions)
gru_predictions = scaler.inverse_transform(gru_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Affichage des résultats
st.subheader("Comparaison des prédictions")

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test_actual.flatten(), name='Valeurs réelles'))
fig.add_trace(go.Scatter(y=cnn_predictions.flatten(), name='Prédictions CNN1D'))
fig.add_trace(go.Scatter(y=gru_predictions.flatten(), name='Prédictions GRU'))
st.plotly_chart(fig)

# Prédiction des 3 prochaines semaines
st.subheader("Prédiction pour les 3 prochaines semaines")
last_sequence = X_test[-1].reshape(1, lookback, 1)
future_predictions = []
current_sequence = last_sequence.copy()

for _ in range(15):  # 15 jours = 3 semaines
    cnn_pred = cnn_model.predict(current_sequence)
    gru_pred = gru_model.predict(current_sequence)
    
    future_predictions.append({
        'CNN1D': float(scaler.inverse_transform(cnn_pred)[0][0]),
        'GRU': float(scaler.inverse_transform(gru_pred)[0][0])
    })
    
    # Mise à jour de la séquence pour la prochaine prédiction
    current_sequence = np.roll(current_sequence, -1, axis=1)
    current_sequence[0, -1, 0] = cnn_pred[0][0]  # Utilisation de la prédiction CNN comme entrée

future_df = pd.DataFrame(future_predictions)
future_df.index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=15)

st.line_chart(future_df)

# Affichage des métriques
st.subheader("Métriques de performance")
cnn_mse = np.mean((cnn_predictions - y_test_actual) ** 2)
gru_mse = np.mean((gru_predictions - y_test_actual) ** 2)

st.write(f"MSE CNN1D: {cnn_mse:.4f}")
st.write(f"MSE GRU: {gru_mse:.4f}")