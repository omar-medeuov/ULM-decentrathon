import h3
import pandas as pd

def analyze_geodata(df):
    df['h3_index'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lng'], 9), axis=1)
    return df