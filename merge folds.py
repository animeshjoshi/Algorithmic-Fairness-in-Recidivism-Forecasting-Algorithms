
import pandas as pd
import glob
import os

# Path to your folder of CSVs
folder_path = 'Probability Frames'

# Get all CSV file paths in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Load and concatenate all CSVs
df_all = pd.concat([pd.read_csv(file).drop('Unnamed: 0', axis = 1) for file in csv_files], ignore_index=True)

df_all.to_csv('Probability Folds Merged.csv')

# Optional: Preview the result
print(df_all.head())