import torch
import torch.nn as nn
import os
from models.H3model import H3model
from src.data.load_data import load_data
from src.data.h3_analyze import (
    analyze_geodata, plot_heatmap_scatterplot, plot_heatmap_on_map, optimize_driver_distribution
)
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd

# 1. Load and analyze data
df = load_data()
df = analyze_geodata(df)

# 2. Aggregate demand per H3 cell
h3_features = df.groupby('h3_index').agg({
    'lat': 'mean',
    'lng': 'mean',
    'alt': 'mean',
    'spd': 'mean',
    'azm': 'mean',
    'randomized_id': 'count'
}).reset_index()
h3_features = h3_features.rename(columns={'randomized_id': 'demand'})

# 3. Prepare features and target
X = h3_features[['lat', 'lng', 'alt', 'spd', 'azm']].values
y = h3_features['demand'].values

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = H3model(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X_train))
    loss = criterion(outputs, torch.FloatTensor(y_train).view(-1, 1))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Predict demand for all H3 cells
model.eval()
with torch.no_grad():
    predicted_demand = model(torch.FloatTensor(X)).numpy().flatten()
h3_features['predicted_demand'] = predicted_demand

# 7. Save processed data
processed_info_dir = "data/processed"
os.makedirs(processed_info_dir, exist_ok=True)
h3_features.to_csv(os.path.join(processed_info_dir, "processed_h3_data.csv"), index=False)
print("Processed H3 data was saved to data/processed/processed_h3_data.csv")

# 8. Plot heatmap (using predicted demand)
h3_features['h3_index'] = h3_features['h3_index'].astype(str)
df_heatmap = h3_features.copy()
df_heatmap['count'] = df_heatmap['predicted_demand']
plot_heatmap_scatterplot(df_heatmap, save_path=os.path.join(processed_info_dir, "demand_heatmap.png"))
plot_heatmap_on_map(df_heatmap, save_path=os.path.join(processed_info_dir, "demand_heatmap_MAP.html"))

# 9. Allocate drivers based on predicted demand
def allocate_drivers(h3_df, total_drivers=100, min_drivers_per_cell=0, demand_threshold=0):
    h3_df = h3_df.copy()
    # Filter out cells with very low predicted demand
    h3_df = h3_df[h3_df['predicted_demand'] > demand_threshold]
    # Proportional allocation
    h3_df['driver_allocation'] = (
        h3_df['predicted_demand'] / h3_df['predicted_demand'].sum() * total_drivers
    ).round().astype(int)
    # Ensure minimum drivers per cell if needed
    if min_drivers_per_cell > 0:
        h3_df.loc[h3_df['driver_allocation'] < min_drivers_per_cell, 'driver_allocation'] = min_drivers_per_cell
    # Rebalance if total exceeds total_drivers
    total_alloc = h3_df['driver_allocation'].sum()
    if total_alloc > total_drivers:
        # Reduce from largest allocations
        diff = total_alloc - total_drivers
        for idx in h3_df.sort_values('driver_allocation', ascending=False).index:
            if diff == 0:
                break
            if h3_df.at[idx, 'driver_allocation'] > min_drivers_per_cell:
                h3_df.at[idx, 'driver_allocation'] -= 1
                diff -= 1
    print(h3_df[['h3_index', 'predicted_demand', 'driver_allocation']].sort_values('driver_allocation', ascending=False).head(10))
    return h3_df

allocate_drivers(h3_features, total_drivers=100)