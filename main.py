import torch
import torch.nn as nn
import os
from models.H3model import H3model
from src.data.load_data import load_data
from src.data.preprocess import (preprocess_data)
from src.data.h3_analyze import (analyze_geodata, identify_popular_routes, plot_heatmap, optimize_driver_distribution, safety_scenarios)
from sklearn.model_selection import *
import torch.optim as optim

# Загрузка и анализ данных
df = load_data()
df = analyze_geodata(df)
X = preprocess_data(df)

popular_routes = identify_popular_routes(df)
print("Popular routes:\n ", popular_routes.head())

unusual = safety_scenarios(df)
print(unusual[['randomized_id', 'lat', 'lng', 'spd', 'alt', 'azm', 'reason']].head())

# Разделение данных
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Создание и обучение модели
model = H3model(input_size=X_train.shape[1])
criterion = nn.MSELoss()  # Для регрессии
optimizer = optim.Adam(model.parameters(), lr=0.001)

y = df['spd'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
outputs = model(torch.FloatTensor(X_train))


# Обучение модели
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Обнуление градиентов

    # Прямое распространение
    outputs = model(torch.FloatTensor(X_train))  # Преобразование данных в тензоры
    loss = criterion(outputs, torch.FloatTensor(y_train).view(-1, 1))  # Вычисление потерь

    # Обратное распространение
    loss.backward()  # Вычисление градиентов
    optimizer.step()  # Обновление весов

    # Вывод информации о процессе обучения
    if (epoch + 1) % 10 == 0:  # Каждые 10 эпох
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

processed_info_dir = "data/processed"
df.to_csv(os.path.join(processed_info_dir, "processed_data.csv"), index=False)
print("Processed data was saved to data/processed/processed_data.csv")
plot_heatmap(df, save_path=os.path.join(processed_info_dir, "demand_heatmap.png"))