import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df.dropna(inplace=True)  # удаление строк с пропусками
    # удаление ненужных столбцов
    df = df[['lat', 'lng', 'alt', 'spd', 'azm']]  # оставляем только нужные столбцы
    scaler = StandardScaler()
    X = scaler.fit_transform(df)  # нормализация данных
    return X