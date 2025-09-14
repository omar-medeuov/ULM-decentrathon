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

def plot_od_on_map(df, save_path=None, top_n=50):
    """
    Визуализация OD-матрицы на карте.
    
    df : DataFrame
        Должен содержать колонки start_h3_index, end_h3_index, count
    save_path : str
        Куда сохранить HTML-карту
    top_n : int
        Количество самых популярных маршрутов для отображения
    """

    # отбираем топ-N популярных маршрутов
    routes = (
        df.groupby(['start_h3_index', 'end_h3_index'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)
    )

    # центр карты = среднее значение по всем маршрутам
    routes['start_lat'] = routes['start_h3_index'].apply(lambda h: h3.cell_to_latlng(h)[0])
    routes['start_lng'] = routes['start_h3_index'].apply(lambda h: h3.cell_to_latlng(h)[1])
    routes['end_lat']   = routes['end_h3_index'].apply(lambda h: h3.cell_to_latlng(h)[0])
    routes['end_lng']   = routes['end_h3_index'].apply(lambda h: h3.cell_to_latlng(h)[1])

    center_lat = routes[['start_lat','end_lat']].to_numpy().mean()
    center_lng = routes[['start_lng','end_lng']].to_numpy().mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # рисуем линии между start и end
    for _, row in routes.iterrows():
        folium.PolyLine(
            locations=[(row['start_lat'], row['start_lng']),
                       (row['end_lat'], row['end_lng'])],
            weight=2 + (row['count'] / routes['count'].max()) * 5,  # толщина зависит от спроса
            color="blue",
            opacity=0.6,
            tooltip=f"Count: {row['count']}"
        ).add_to(m)

    if save_path:
        m.save(save_path)
        print(f"OD map saved to {save_path}")

    return m

def identify_popular_routes(df):
    popular_routes = df.groupby(['start_h3_index', 'end_h3_index']).size().reset_index(name='count')
    return popular_routes.sort_values(by='count', ascending=False)


def plot_heatmap_on_map(df, value_column="count", save_path=None, top_quantile=0.6):
    """
    Строит интерактивную тепловую карту (folium) только для топовых зон.
    
    top_quantile: float
        Порог по квантилю (0.7 = оставить только 30% самых "горячих" зон)
    """
    if "lat" not in df.columns or "lng" not in df.columns:
        df = df.copy()
        df['lat'] = df['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[0])
        df['lng'] = df['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[1])

    # агрегируем по h3
    h3_counts = df.groupby("h3_index")[value_column].sum().reset_index()
    h3_counts['lat'] = h3_counts['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[0])
    h3_counts['lng'] = h3_counts['h3_index'].apply(lambda h: h3.cell_to_latlng(h)[1])

    # фильтруем только топовые ячейки
    threshold = h3_counts[value_column].quantile(top_quantile)
    h3_top = h3_counts[h3_counts[value_column] >= threshold]

    # центр карты
    center_lat = h3_top['lat'].mean()
    center_lng = h3_top['lng'].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)

    # нормализация значений (чтобы цвета были адекватные)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    h3_top['normalized'] = scaler.fit_transform(h3_top[[value_column]])

    heat_data = h3_top[['lat', 'lng', 'normalized']].values.tolist()
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
