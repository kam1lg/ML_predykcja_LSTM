import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Połączenie z bazą danych MySQL i wczytanie danych
engine = create_engine('mysql+mysqlconnector://root:@localhost/kryptowaluty')
df = pd.read_sql("SELECT Data, Czas, Rate FROM bch_usd_historyczne ORDER BY Data, Czas", engine)
df['Data'] = df['Data'].astype(str)
df['Czas'] = df['Czas'].astype(str)
data = df['Rate'].values.reshape(-1, 1)

# Skalowanie danych do zakresu od 0 do 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Podział danych na zestawy treningowe i testowe
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Tworzenie zbioru danych z sekwencjami do modelu LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Przekształcanie danych, aby pasowały do wymagań wejściowych LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Budowanie modelu LSTM
model_bch = Sequential()
model_bch.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model_bch.add(LSTM(50, return_sequences=False))
model_bch.add(Dense(25))
model_bch.add(Dense(1))

# Kompilacja i trenowanie modelu LSTM z wczesnym zatrzymaniem
model_bch.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_bch.fit(X_train, y_train, validation_split=0.2, batch_size=1, epochs=20, callbacks=[early_stopping])
model_bch.save('bch_usd_lstm_model.h5')
