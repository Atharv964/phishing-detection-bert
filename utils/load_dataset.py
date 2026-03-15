import pandas as pd

# load dataset
data = pd.read_csv("dataset/raw/legitphish.csv")

# show first rows
print("First 5 rows:\n")
print(data.head())

# show dataset info
print("\nDataset Info:\n")
print(data.info())

# check class distribution
print("\nClass Distribution:\n")
print(data["ClassLabel"].value_counts())