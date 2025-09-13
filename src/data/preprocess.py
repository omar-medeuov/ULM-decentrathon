import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df.dropna(inplace=True)  # Удаление строк с пропусками
    # Удаление ненужных столбцов, если необходимо
    df = df[['lat', 'lng', 'alt', 'spd', 'azm']]  # Оставляем только нужные столбцы
    scaler = StandardScaler()
    X = scaler.fit_transform(df)  # Нормализация данных
    return X