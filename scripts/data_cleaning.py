import pandas as pd
import os

# Load raw data
input_path = os.path.join('..', 'data', 'raw', 'insurance.csv')
output_path = os.path.join('..', 'data', 'processed', 'cleaned.csv')

df = pd.read_csv(input_path)

# 🔁 Minimal Cleaning: Do NOT drop rows!
# df.dropna(inplace=True)  ❌ COMMENT THIS OUT

# Encode target column
df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

# Optional feature engineering
df['claim_ratio'] = df['total_claim_amount'] / (df['age'] + 1e-5)

# Save cleaned data
df.to_csv(output_path, index=False)

print("✅ Cleaned data saved with shape:", df.shape)
