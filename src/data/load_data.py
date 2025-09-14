import pandas as pd
import yaml

def load_data():
    # загрузка конфига
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    csv_path = config['data']['csv_path']
    df = pd.read_csv(csv_path)
    return df