from h3 import geo_to_h3
import pandas as pd
import seaborn as sns
import matplotlib as plt

def analyze_geodata(df):
    # Assign H3 index for each point
    df['h3_index'] = df.apply(lambda row: geo_to_h3(row['lat'], row['lng'], 9), axis=1)
    # For each trip, get start and end H3 index
    df = df.sort_values(['randomized_id', 'azm'])  # Sort by trip and azimuth (or timestamp if available)
    start_h3 = df.groupby('randomized_id')['h3_index'].first().rename('start_h3_index')
    end_h3 = df.groupby('randomized_id')['h3_index'].last().rename('end_h3_index')
    df = df.join(start_h3, on='randomized_id')
    df = df.join(end_h3, on='randomized_id')
    return df

def identify_popular_routes(df):
    # Группировка по маршрутам и подсчет количества поездок
    popular_routes = df.groupby(['start_h3_index', 'end_h3_index']).size().reset_index(name='count')
    return popular_routes.sort_values(by='count', ascending=False)

def plot_heatmap(df):
    # Count points per (lat, lng)
    df['count'] = 1
    heatmap_data = df.pivot_table(index='lat', columns='lng', values='count', aggfunc='sum', fill_value=0)
    sns.heatmap(heatmap_data, cmap='YlGnBu')
    plt.title('Тепловая карта спроса')
    plt.show()

def optimize_driver_distribution(df, total_drivers=100):
    demand = df.groupby('h3_index').size().reset_index(name='demand')
    demand['driver_allocation'] = (demand['demand'] / demand['demand'].sum() * total_drivers).round().astype(int)
    print(demand[['h3_index', 'demand', 'driver_allocation']].sort_values('driver_allocation', ascending=False))
    return demand

def safety_scenarios(
    df,
    threshold_speed=120,
    threshold_alt=2000,
    azm_change_threshold=90,
    rare_route_quantile=0.05
):
    """
    Выявляет:
    - Слишком высокую скорость или высоту
    - Резкие изменения курса (азимута)
    - Редкие маршруты (непопулярные start-end H3 пары)
    """
    # 1. Необычная скорость или высота
    unusual_physical = df[(df['spd'] > threshold_speed) | (df['alt'] > threshold_alt)].copy()
    unusual_physical['reason'] = 'speed/altitude'

    # 2. Резкая смена курса (азимута)
    df_sorted = df.sort_values(['randomized_id', 'azm'])
    df_sorted['azm_prev'] = df_sorted.groupby('randomized_id')['azm'].shift(1)
    df_sorted['azm_diff'] = (df_sorted['azm'] - df_sorted['azm_prev']).abs()
    sudden_turns = df_sorted[df_sorted['azm_diff'] > azm_change_threshold].copy()
    sudden_turns['reason'] = 'azimuth_change'

    # 3. Редкие маршруты
    route_counts = df.groupby(['start_h3_index', 'end_h3_index']).size().reset_index(name='count')
    rare_threshold = route_counts['count'].quantile(rare_route_quantile)
    rare_routes = route_counts[route_counts['count'] <= rare_threshold][['start_h3_index', 'end_h3_index']]
    rare_route_points = df.merge(rare_routes, on=['start_h3_index', 'end_h3_index'], how='inner').copy()
    rare_route_points['reason'] = 'rare_route'

    # Объединяем все необычные случаи
    unusual = pd.concat([unusual_physical, sudden_turns, rare_route_points], ignore_index=True)
    unusual = unusual.drop_duplicates()
    return unusual