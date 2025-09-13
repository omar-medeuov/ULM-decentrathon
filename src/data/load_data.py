import pandas as pd

file_path = "data/raw/raw_dataset.csv" # <- csv path 
df = pd.read_csv(file_path)
print(df.head())