import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import schedule
import time

def predict_next_hour(model_file, input_table, output_table, code):
    # Połączenie z bazą danych MySQL i wczytanie danych
    engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')
    connection = engine.connect()
    df = pd.read_sql(
        f"SELECT Data, TIME_FORMAT(Czas, '%H:%i:%s') AS Czas, Rate FROM {input_table} ORDER BY Data DESC, Czas DESC LIMIT 60",
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

    # Wstawienie przewidywanych danych do bazy danych
    cursor = connection.connection.cursor()
    sql = f"INSERT INTO {output_table} (Data, Czas, Code, Rate) VALUES (%s, %s, %s, %s)"
    values = (next_date_str, next_time_str, code, float(prediction[0][0]))  # Konwersja na float
    cursor.execute(sql, values)
    connection.connection.commit()
    cursor.close()

    connection.close()
    engine.dispose()

# Funkcja planująca wywoływanie predict_next_hour dla wszystkich walut co 60 minut
def schedule_tasks():
    predict_next_hour('btc_usd_lstm_model.h5', 'btc_usd_historyczne', 'btc_usd_predykcja', 'BTC/USD')
    predict_next_hour('bch_usd_lstm_model.h5', 'bch_usd_historyczne', 'bch_usd_predykcja', 'BCH/USD')
    predict_next_hour('eth_usd_lstm_model.h5', 'eth_usd_historyczne', 'eth_usd_predykcja', 'ETH/USD')
    predict_next_hour('ltc_usd_lstm_model.h5', 'ltc_usd_historyczne', 'ltc_usd_predykcja', 'LTC/USD')
    predict_next_hour('xrp_usd_lstm_model.h5', 'xrp_usd_historyczne', 'xrp_usd_predykcja', 'XRP/USD')
    schedule.every(60).minutes.do(predict_next_hour, 'btc_usd_lstm_model.h5', 'btc_usd_historyczne', 'btc_usd_predykcja', 'BTC/USD')
    schedule.every(60).minutes.do(predict_next_hour, 'bch_usd_lstm_model.h5', 'bch_usd_historyczne', 'bch_usd_predykcja', 'BCH/USD')
    schedule.every(60).minutes.do(predict_next_hour, 'eth_usd_lstm_model.h5', 'eth_usd_historyczne', 'eth_usd_predykcja', 'ETH/USD')
    schedule.every(60).minutes.do(predict_next_hour, 'ltc_usd_lstm_model.h5', 'ltc_usd_historyczne', 'ltc_usd_predykcja', 'LTC/USD')
    schedule.every(60).minutes.do(predict_next_hour, 'xrp_usd_lstm_model.h5', 'xrp_usd_historyczne', 'xrp_usd_predykcja', 'XRP/USD')

if __name__ == "__main__":
    schedule_tasks()
    while True:
        schedule.run_pending()
        time.sleep(1)
