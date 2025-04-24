import pandas as pd

# Load the CSV file
df = pd.read_csv("/home/jiw220/acoustic-multiclass-training/132PeruXC_RawChunks_Fixed.csv")

# Count distinct values in a specific column (replace 'column_name' with the actual column name)
unique_count = df['SPECIES'].nunique()

print(f"Number of distinct values in 'column_name': {unique_count}")
