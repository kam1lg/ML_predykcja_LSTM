import csv
import mysql.connector
from datetime import datetime

# Funkcja do wczytywania danych z pliku CSV i wstawiania ich do tabeli w bazie danych MySQL
def insert_data(file_path, table_name):
    # Nawiązanie połączenia z bazą danych MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="kryptowaluty"
    )
    cursor = conn.cursor()
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            unix_timestamp = int(row[0])
            date_time_str = datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            date_str, time_str = date_time_str.split(' ')
            code = row[2]
            rate = float(row[6])
            sql = f"INSERT INTO {table_name} (Data, Czas, Code, Rate) VALUES (%s, %s, %s, %s)"
            values = (date_str, time_str, code, rate)
            cursor.execute(sql, values)
    conn.commit()
    cursor.close()
    conn.close()

# Wczytywanie danych dla różnych kryptowalut
insert_data('Bitstamp_BTCUSD_1h.csv', 'btc_usd_historyczne')
insert_data('Bitstamp_ETHUSD_1h.csv', 'eth_usd_historyczne')
insert_data('Bitstamp_LTCUSD_1h.csv', 'ltc_usd_historyczne')
insert_data('Bitstamp_XRPUSD_1h.csv', 'xrp_usd_historyczne')
insert_data('Bitstamp_BCHUSD_1h.csv', 'bch_usd_historyczne')
