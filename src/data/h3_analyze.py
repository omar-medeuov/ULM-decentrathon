from h3 import latlng_to_cell
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import h3
import folium
from folium.plugins import HeatMap

# Алгоритм анализа H3 --->

def analyze_geodata(df):
    # <--- Назначение H3 идекса на каждую точку --->
    df['h3_index'] = df.apply(lambda row: latlng_to_cell(row['lat'], row['lng'], 9), axis=1)
    # <--- For each trip, get start and end H3 index --->
    df = df.sort_values(['randomized_id', 'azm'])  # <--- Sort by trip and azimuth --->
    start_h3 = df.groupby('randomized_id')['h3_index'].first().rename('start_h3_index')
    end_h3 = df.groupby('randomized_id')['h3_index'].last().rename('end_h3_index')
    df = df.join(start_h3, on='randomized_id')
    df = df.join(end_h3, on='randomized_id')
    return df

def identify_popular_routes(df):
    popular_routes = df.groupby(['start_h3_index', 'end_h3_index']).size().reset_index(name='count')
    return popular_routes.sort_values(by='count', ascending=False)

def plot_heatmap_scatterplot(df, save_path=None):
    # <--- Aggregate demand by H3 cell --->
    h3_counts = df['h3_index'].value_counts().reset_index()
    h3_counts.columns = ['h3_index', 'count']
    # Get the center lat/lng for each H3 cell
    import h3
    h3_counts['lat'] = h3_counts['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[0])
    h3_counts['lng'] = h3_counts['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[1])
    # <--- Plot using rounded coordinates для тепловой карты!!! --->
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=h3_counts, x='lng', y='lat', size='count', hue='count',
        palette='YlGnBu', legend=False, sizes=(20, 200)
    )
    plt.title('Тепловая карта спроса (по H3)')
    plt.xlim(h3_counts['lng'].min() - 0.01, h3_counts['lng'].max() + 0.01)
    plt.ylim(h3_counts['lat'].min() - 0.01, h3_counts['lat'].max() + 0.01)
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    plt.show()


def plot_heatmap_on_map(df, save_path=None):
    h3_counts = df['h3_index'].value_counts().reset_index()
    h3_counts.columns = ['h3_index', 'count']
    h3_counts['lat'] = h3_counts['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[0])
    h3_counts['lng'] = h3_counts['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[1])
    center_lat = h3_counts['lat'].mean()
    center_lng = h3_counts['lng'].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # ограничение карты, чтобы не включать всю карту мира
    sw = [h3_counts['lat'].min(), h3_counts['lng'].min()]
    ne = [h3_counts['lat'].max(), h3_counts['lng'].max()]
    m.fit_bounds([sw, ne])

    heat_data = h3_counts[['lat', 'lng', 'count']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

    if save_path:
        m.save(save_path)
        print(f"Interactive heatmap saved to {save_path}")
    return m

def optimize_driver_distribution(df, total_drivers=100):
    demand = df.groupby('h3_index').size().reset_index(name='demand')
    demand['driver_allocation'] = (demand['demand'] / demand['demand'].sum() * total_drivers).round().astype(int)
    print(demand[['h3_index', 'demand', 'driver_allocation']].sort_values('driver_allocation', ascending=False))
    return demand

# алгоритм для сценария безопасности (отклонения от маршрута, скорость, и тд...) 
def safety_scenarios(
    df,
    threshold_speed=120,
    threshold_alt=2000,
    azm_change_threshold=90,
    rare_route_quantile=0.05
):
    unusual_physical = df[(df['spd'] > threshold_speed) | (df['alt'] > threshold_alt)].copy()
    unusual_physical['reason'] = 'speed/altitude'

    df_sorted = df.sort_values(['randomized_id', 'azm'])
    df_sorted['azm_prev'] = df_sorted.groupby('randomized_id')['azm'].shift(1)
    df_sorted['azm_diff'] = (df_sorted['azm'] - df_sorted['azm_prev']).abs()
    sudden_turns = df_sorted[df_sorted['azm_diff'] > azm_change_threshold].copy()
    sudden_turns['reason'] = 'azimuth_change'

    route_counts = df.groupby(['start_h3_index', 'end_h3_index']).size().reset_index(name='count')
    rare_threshold = route_counts['count'].quantile(rare_route_quantile)
    rare_routes = route_counts[route_counts['count'] <= rare_threshold][['start_h3_index', 'end_h3_index']]
    rare_route_points = df.merge(rare_routes, on=['start_h3_index', 'end_h3_index'], how='inner').copy()
    rare_route_points['reason'] = 'rare_route'

    unusual = pd.concat([unusual_physical, sudden_turns, rare_route_points], ignore_index=True)
    unusual = unusual.drop_duplicates()
    return unusual
