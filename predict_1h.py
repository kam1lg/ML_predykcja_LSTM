import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta


def predict_next_hour(model_file, table_name, code):
    # Połączenie z bazą danych MySQL i wczytanie danych
    engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')
    connection = engine.connect()
    df = pd.read_sql(
        f"SELECT Data, TIME_FORMAT(Czas, '%H:%i:%s') AS Czas, Rate FROM {table_name} ORDER BY Data DESC, Czas DESC LIMIT 60",
        connection)

    # Odwrócenie kolejności, aby najnowsze dane były na końcu
    df = df.iloc[::-1]
    data = df['Rate'].values.reshape(-1, 1)

    # Skalowanie danych
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Przygotowanie danych do predykcji
    X = np.reshape(scaled_data, (1, scaled_data.shape[0], 1))

    # Ładowanie modelu i dokonywanie predykcji
    model = load_model(model_file)
    prediction = model.predict(X)
    prediction = scaler.inverse_transform(prediction)

    # Obliczenie daty i czasu dla predykcji
    last_date_time_str = f"{df['Data'].iloc[-1]} {df['Czas'].iloc[-1]}"
    last_date_time = datetime.strptime(last_date_time_str, '%Y-%m-%d %H:%M:%S')
    next_date_time = last_date_time + timedelta(hours=1)
    next_date_str = next_date_time.strftime('%Y-%m-%d')
    next_time_str = next_date_time.strftime('%H:%M:%S')

    # Wyświetlenie wyniku
    print(f"Predykcja dla {code} na {next_date_str} {next_time_str}: {prediction[0][0]}")

    connection.close()
    engine.dispose()


# Przykład użycia
predict_next_hour('btc_usd_lstm_model.h5', 'btc_usd_historyczne', 'BTC/USD')
predict_next_hour('bch_usd_lstm_model.h5', 'bch_usd_historyczne', 'BCH/USD')
predict_next_hour('eth_usd_lstm_model.h5', 'eth_usd_historyczne', 'ETH/USD')
predict_next_hour('ltc_usd_lstm_model.h5', 'ltc_usd_historyczne', 'LTC/USD')
predict_next_hour('xrp_usd_lstm_model.h5', 'xrp_usd_historyczne', 'XRP/USD')
