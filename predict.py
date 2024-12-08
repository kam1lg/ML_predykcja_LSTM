import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model

# Funkcja do wczytywania danych z bazy danych MySQL
def load_data(query):
    engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')  # Tworzenie połączenia z bazą danych
    df = pd.read_sql(query, engine)  # Wykonanie zapytania SQL i załadowanie danych do DataFrame
    df['Data'] = df['Data'].astype(str)  # Konwersja kolumny 'Data' na typ string
    df['Czas'] = df['Czas'].astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})')[0]  # Ekstrakcja czasu z kolumny 'Czas'
    df['DateTime'] = pd.to_datetime(df['Data'] + ' ' + df['Czas'], format='%Y-%m-%d %H:%M:%S')  # Utworzenie kolumny 'DateTime'
    return df

# Funkcja do tworzenia zbioru danych dla modelu LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])  # Tworzenie sekwencji wejściowych
        Y.append(dataset[i + look_back, 0])  # Tworzenie wartości wyjściowych
    return np.array(X), np.array(Y)

# Funkcja do wypełniania brakujących danych za pomocą modelu LSTM
def fill_missing_data(df, model, scaler, look_back):
    date_range = pd.date_range(start=df['DateTime'].min(), end=df['DateTime'].max(), freq='H')  # Zakres dat z godzinnymi odstępami
    df_full = pd.DataFrame(date_range, columns=['DateTime'])  # Tworzenie DataFrame z pełnym zakresem dat
    df_full = df_full.merge(df, on='DateTime', how='left')  # Łączenie pełnego zakresu dat z rzeczywistymi danymi

    for i in range(len(df_full)):
        if pd.isnull(df_full.iloc[i]['Rate']):  # Sprawdzenie, czy wartość jest pusta
            input_data = df_full.iloc[i-look_back:i]['Rate'].values  # Pobranie danych wejściowych dla modelu
            input_data = input_data.reshape(-1, 1)
            input_data_scaled = scaler.transform(input_data)  # Skalowanie danych wejściowych
            input_data_scaled = input_data_scaled.reshape(1, look_back, 1)
            predicted_rate = model.predict(input_data_scaled)  # Przewidywanie brakującej wartości
            predicted_rate = scaler.inverse_transform(predicted_rate)
            df_full.at[df_full.index[i], 'Rate'] = predicted_rate[0, 0]  # Wypełnianie brakującej wartości przewidywaną

    return df_full

# Funkcja do zapisywania przewidywań do bazy danych
def save_predictions_to_db(df, table_name):
    engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')  # Tworzenie połączenia z bazą danych
    df['Data'] = df['DateTime'].dt.date
    df['Czas'] = df['DateTime'].dt.time
    df[['Data', 'Czas', 'Rate']].to_sql(table_name, engine, if_exists='replace', index=False)  # Zapisywanie do bazy danych

look_back = 60  # Długość sekwencji wejściowych

# Funkcja do przewidywania i zapisywania wyników
def predict_and_save(model_path, query, table_name):
    df = load_data(query)  # Wczytanie danych
    scaler = MinMaxScaler(feature_range=(0, 1))  # Tworzenie skalera
    train_size = int(len(df) * 0.8)  # Wyznaczenie rozmiaru zbioru treningowego
    df['Rate'] = scaler.fit_transform(df['Rate'].values.reshape(-1, 1))  # Skalowanie danych
    train_data, test_data = df['Rate'].values[:train_size], df['Rate'].values[train_size:]  # Podział na zbiory treningowe i testowe
    X_train, y_train = create_dataset(train_data, look_back)  # Tworzenie zbioru treningowego
    X_test, y_test = create_dataset(test_data, look_back)  # Tworzenie zbioru testowego
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = load_model(model_path)  # Ładowanie wytrenowanego modelu
    predictions = model.predict(X_test)  # Przewidywanie na zbiorze testowym
    predictions = scaler.inverse_transform(predictions)  # Odwrotne skalowanie przewidywań

    df_full = fill_missing_data(df, model, scaler, look_back)  # Wypełnianie brakujących danych
    save_predictions_to_db(df_full, table_name)  # Zapisywanie wyników do bazy danych

# Przewidywanie dla BTC/USD
predict_and_save('btc_usd_lstm_model.h5',
                 "SELECT Data, Czas, Rate FROM btc_usd_historyczne ORDER BY Data, Czas",
                 'btc_usd_predykcja')

# Przewidywanie dla ETH/USD
predict_and_save('eth_usd_lstm_model.h5',
                 "SELECT Data, Czas, Rate FROM eth_usd_historyczne ORDER BY Data, Czas",
                 'eth_usd_predykcja')

# Przewidywanie dla XLM/USD
predict_and_save('xlm_usd_lstm_model.h5',
                 "SELECT Data, Czas, Rate FROM xlm_usd_historyczne ORDER BY Data, Czas",
                 'xlm_usd_predykcja')
