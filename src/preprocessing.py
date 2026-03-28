
import pandas as pd
import numpy as np
import os


DATA_PATH = "Data/raw/predictive_maintenance.csv"
OUTPUT_PATH = "Data/processed/processed_data.csv"

print("Loading raw data...")
df = pd.read_csv(DATA_PATH)

print("\nFirst 5 rows:") 
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# UDI and Product ID are just identifiers, not useful for prediction
df = df.drop(['UDI', 'Product ID'], axis=1)

print("\nAfter dropping ID columns:")
print(df.columns)


# Type column has L, M, H
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

print("\nAfter encoding 'Type':")
print(df.columns)

# 4. Feature Engineering (Very Important Part)

# Temperature difference (process - air)
df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

# Power feature (physics-based)
df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

# Wear per speed
df['Wear_per_speed'] = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1)

print("\nNew engineered features added:")
print(['Temp_diff', 'Power', 'Wear_per_speed'])


# 5. Drop Failure Type Columns (Optional for now)

df = df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)

print("\nFinal columns used for modeling:")
print(df.columns)


# 6. Check Class Imbalance

print("\nClass distribution (Machine failure):")
print(df['Machine failure'].value_counts())


# 7. Save Processed Dataset

# Create processed folder if not exists
os.makedirs("Data/processed", exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nProcessed data saved successfully at:")
print(OUTPUT_PATH)
print("\nWeek 1 preprocessing completed successfully!")
