import csv
import mysql.connector
from datetime import datetime

# Połączenie z bazą danych
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="kryptowaluty"
)
cursor = conn.cursor()

# Funkcja do wstawiania danych do tabeli kursy_fiat_historyczne
def insert_data(date, time, code, rate):
    sql = "INSERT INTO kursy_fiat_historyczne (data, czas, code, rate) VALUES (%s, %s, %s, %s)"
    values = (date, time, code, rate)
    cursor.execute(sql, values)
    conn.commit()

# Wczytywanie danych z pliku CSV
with open('archiwum_tab_a_2024.csv', mode='r') as file:
    reader = csv.reader(file, delimiter='\t')
    headers = next(reader)[0].split(';')  # Podział nagłówków na komórki

    for row in reader:
        cells = row[0].split(';')
        date_str = cells[0]
        date = datetime.strptime(date_str, '%Y%m%d').date()
        time = "00:00:00"  # Zakładam, że czas nie jest podany w pliku CSV, ustawiam na północ

        for i in range(1, len(cells)):
            if i >= len(headers):
                continue  # Pomijanie dodatkowych komórek

            code = headers[i]
            rate = float(cells[i].replace(',', '.'))

            # Dostosowanie kursów dla specyficznych walut
            if code.startswith('100'):
                rate /= 100
                code = code[3:]
            elif code.startswith('1') and len(code) > 3:
                code = code[1:]

            insert_data(date, time, code, rate)

# Zamknięcie połączenia z bazą danych
cursor.close()
conn.close()
