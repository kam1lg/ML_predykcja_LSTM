import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model

def load_and_predict(model_file, table_name, output_table, code):
    # Połączenie z bazą danych MySQL i wczytanie danych
    engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')
    connection = engine.connect()
    df = pd.read_sql(f"SELECT Data, Czas, Rate FROM {table_name} ORDER BY Data, Czas", connection)
    data = df['Rate'].values.reshape(-1, 1)

    # Skalowanie danych
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Tworzenie zbioru danych do predykcji
    look_back = 60
    def create_dataset(dataset, look_back=1):
        X = []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
        return np.array(X)

    X = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Ładowanie modelu i dokonywanie predykcji
    model = load_model(model_file)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Dodanie przewidywanych danych do DataFrame
    pred_df = df.iloc[look_back:].copy()
    pred_df['Rate'] = predictions
    pred_df['Code'] = code

    # Wstawienie przewidywanych danych do bazy danych za pomocą INSERT INTO
    cursor = connection.connection.cursor()
    for index, row in pred_df.iterrows():
        sql = f"INSERT INTO {output_table} (Data, Czas, Code, Rate) VALUES (%s, %s, %s, %s)"
        values = (row['Data'], row['Czas'], row['Code'], row['Rate'])
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

# Wykonanie predykcji i zapisanie wyników
for model_file, table_name, output_table, code in models_tables:
    load_and_predict(model_file, table_name, output_table, code)
