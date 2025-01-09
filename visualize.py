import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

def plot_data(engine, history_table, prediction_table, title, start_date, end_date):
    # Pobieranie danych historycznych
    df_hist = pd.read_sql(f"SELECT Data, Czas, Rate FROM {history_table} ORDER BY Data, Czas", engine)
    df_hist['Czas'] = df_hist['Czas'].apply(lambda x: str(x).split(' ')[-1])  # Usunięcie "0 days"
    df_hist['Data_Czas'] = pd.to_datetime(df_hist['Data'].astype(str) + ' ' + df_hist['Czas'])

    # Pobieranie danych przewidzianych
    df_pred = pd.read_sql(f"SELECT Data, Czas, Rate FROM {prediction_table} ORDER BY Data, Czas", engine)
    df_pred['Czas'] = df_pred['Czas'].apply(lambda x: str(x).split(' ')[-1])  # Usunięcie "0 days"
    df_pred['Data_Czas'] = pd.to_datetime(df_pred['Data'].astype(str) + ' ' + df_pred['Czas'])

    # Filtracja danych w określonym przedziale czasowym
    df_hist = df_hist[(df_hist['Data_Czas'] >= start_date) & (df_hist['Data_Czas'] <= end_date)]
    df_pred = df_pred[(df_pred['Data_Czas'] >= start_date) & (df_pred['Data_Czas'] <= end_date)]

    # Wykres
    plt.figure(figsize=(14, 7))
    plt.plot(df_hist['Data_Czas'], df_hist['Rate'], label='Wartości historyczne')
    plt.plot(df_pred['Data_Czas'], df_pred['Rate'], label='Wartości przewidziane')
    plt.title(title)
    plt.xlabel('Czas')
    plt.ylabel('Kurs')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    engine = create_engine('mysql+mysqlconnector://root@localhost/kryptowaluty')

    # Definiowanie zakresu dat
    start_date = '2025-01-01'
    end_date = '2025-01-02'

    # Wykresy dla każdej kryptowaluty w określonym przedziale czasowym
    plot_data(engine, 'btc_usd_historyczne', 'btc_usd_predykcja', 'BTC/USD', start_date, end_date)
    plot_data(engine, 'bch_usd_historyczne', 'bch_usd_predykcja', 'BCH/USD', start_date, end_date)
    plot_data(engine, 'eth_usd_historyczne', 'eth_usd_predykcja', 'ETH/USD', start_date, end_date)
    plot_data(engine, 'ltc_usd_historyczne', 'ltc_usd_predykcja', 'LTC/USD', start_date, end_date)
    plot_data(engine, 'xrp_usd_historyczne', 'xrp_usd_predykcja', 'XRP/USD', start_date, end_date)

if __name__ == '__main__':
    main()
