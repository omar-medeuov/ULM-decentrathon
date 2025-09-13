import pandas as pd

file_path = "data/raw/geo_locations_astana_hackathon.csv"
df = pd.read_csv(file_path)
print(df.head())