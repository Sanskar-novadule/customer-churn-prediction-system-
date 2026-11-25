import pandas as pd

from config import DATA_PATH


# Load dataset
df = pd.read_csv(DATA_PATH)


print("===== DATA LOADED =====")

print("\nFirst 5 rows:")
print(df.head())


print("\nShape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nData Information:")
df.info()