import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Funkcja do wczytywania danych z bazy danych MySQL
def load_data(query):
    engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')
    df = pd.read_sql(query, engine)
    df['Data'] = df['Data'].astype(str)  # Konwersja do string
    df['Czas'] = df['Czas'].astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
    df['DateTime'] = pd.to_datetime(df['Data'] + ' ' + df['Czas'], format='%Y-%m-%d %H:%M:%S')
    return df

# Funkcja do wizualizacji rzeczywistych i przewidywanych danych
def visualize_results(actual_df, pred_df, title, start_date, end_date):
    # Filtrowanie danych według zakresu dat
    mask_actual = (actual_df['DateTime'] >= start_date) & (actual_df['DateTime'] <= end_date)
    mask_pred = (pred_df['DateTime'] >= start_date) & (pred_df['DateTime'] <= end_date)
    actual_filtered = actual_df.loc[mask_actual]
    pred_filtered = pred_df.loc[mask_pred]

    # Tworzenie wykresu
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Kurs Zamknięcia USD', fontsize=18)
    plt.plot(actual_filtered['DateTime'], actual_filtered['Rate'], label='Rzeczywiste')
    plt.plot(pred_filtered['DateTime'], pred_filtered['Rate'], label='Przewidziane')
    plt.legend(loc='lower right')
    plt.show()

start_date = '2024-11-01'
end_date = '2024-11-30'

# Wizualizacja dla BTC/USD
actual_btc = load_data("SELECT Data, Czas, Rate FROM btc_usd_historyczne ORDER BY Data, Czas")
pred_btc = load_data("SELECT Data, Czas, Rate FROM btc_usd_predykcja ORDER BY Data, Czas")
visualize_results(actual_btc, pred_btc, 'Model BTC/USD', start_date, end_date)

# Wizualizacja dla ETH/USD
actual_eth = load_data("SELECT Data, Czas, Rate FROM eth_usd_historyczne ORDER BY Data, Czas")
pred_eth = load_data("SELECT Data, Czas, Rate FROM eth_usd_predykcja ORDER BY Data, Czas")
visualize_results(actual_eth, pred_eth, 'Model ETH/USD', start_date, end_date)

# Wizualizacja dla XLM/USD
actual_xlm = load_data("SELECT Data, Czas, Rate FROM xlm_usd_historyczne ORDER BY Data, Czas")
pred_xlm = load_data("SELECT Data, Czas, Rate FROM xlm_usd_predykcja ORDER BY Data, Czas")
visualize_results(actual_xlm, pred_xlm, 'Model XLM/USD', start_date, end_date)
