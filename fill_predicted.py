import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta


def load_and_predict_continuous(model_file, input_table, output_table, code):
    # Połączenie z bazą danych MySQL
    engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')
    connection = engine.connect()

    # Pobranie ostatniego przewidywanego wpisu
    last_pred = pd.read_sql(f"SELECT Data, Czas FROM {output_table} ORDER BY Data DESC, Czas DESC LIMIT 1", connection)
    if last_pred.empty:
        last_pred = pd.read_sql(f"SELECT Data, Czas FROM {input_table} ORDER BY Data DESC, Czas DESC LIMIT 1",
                                connection)
    last_date_time_str = f"{last_pred['Data'].iloc[0]} {last_pred['Czas'].iloc[0]}"

    # Debugowanie
    print(f"last_date_time_str: {last_date_time_str}")

    try:
        last_date_time = datetime.strptime(last_date_time_str, '%Y-%m-%d 0 days %H:%M:%S')
    except ValueError as e:
        print(f"ValueError: {e}")
        # Spróbujmy alternatywny format
        last_date_time = datetime.strptime(last_date_time_str, '%Y-%m-%d %H:%M:%S')

    # Pobranie ostatnich 60 wpisów z tabeli wejściowej
    df = pd.read_sql(
        f"SELECT Data, TIME_FORMAT(Czas, '%H:%i:%s') AS Czas, Rate FROM {input_table} ORDER BY Data DESC, Czas DESC LIMIT 60",
        connection)

    df = df.iloc[::-1]
    data = df['Rate'].values.reshape(-1, 1)

    # Skalowanie danych
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Przygotowanie danych do predykcji
    X = np.reshape(scaled_data, (1, scaled_data.shape[0], 1))

    # Ładowanie modelu
    model = load_model(model_file)

    # Dokonywanie predykcji co godzinę
    current_time = datetime.now()
    while last_date_time < current_time:
        prediction = model.predict(X)
        prediction_unscaled = scaler.inverse_transform(prediction)

        # Aktualizacja danych wejściowych
        scaled_data = np.append(scaled_data[1:], prediction)
        X = np.reshape(scaled_data, (1, scaled_data.shape[0], 1))

        # Aktualizacja czasu
        last_date_time += timedelta(hours=1)
        next_date_str = last_date_time.strftime('%Y-%m-%d')
        next_time_str = last_date_time.strftime('%H:%M:%S')

        # Sprawdzenie, czy wpis już istnieje w bazie danych
        existing_entry = pd.read_sql(
            f"SELECT 1 FROM {output_table} WHERE Data = '{next_date_str}' AND Czas = '{next_time_str}' LIMIT 1",
            connection)

        if existing_entry.empty:
            # Wstawienie przewidywanych danych do bazy danych
            cursor = connection.connection.cursor()
            sql = f"INSERT INTO {output_table} (Data, Czas, Code, Rate) VALUES (%s, %s, %s, %s)"
            values = (next_date_str, next_time_str, code, float(prediction_unscaled[0][0]))  # Konwersja na float
            cursor.execute(sql, values)
            connection.connection.commit()
            cursor.close()

    connection.close()
    engine.dispose()


# Lista modeli i odpowiadające im tabele
models_tables = [
    ('btc_usd_lstm_model.h5', 'btc_usd_historyczne', 'btc_usd_predykcja', 'BTC/USD'),
    ('bch_usd_lstm_model.h5', 'bch_usd_historyczne', 'bch_usd_predykcja', 'BCH/USD'),
    ('eth_usd_lstm_model.h5', 'eth_usd_historyczne', 'eth_usd_predykcja', 'ETH/USD'),
    ('ltc_usd_lstm_model.h5', 'ltc_usd_historyczne', 'ltc_usd_predykcja', 'LTC/USD'),
    ('xrp_usd_lstm_model.h5', 'xrp_usd_historyczne', 'xrp_usd_predykcja', 'XRP/USD')
]

# Wykonanie ciągłych predykcji i zapisanie wyników
for model_file, input_table, output_table, code in models_tables:
    load_and_predict_continuous(model_file, input_table, output_table, code)
