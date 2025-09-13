import pandas as pd

file_path = "" # <- csv path 
df = pd.read_csv(file_path)
print(df.head())