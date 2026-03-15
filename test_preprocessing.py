from utils.preprocessing import load_and_clean_data

data = load_and_clean_data("dataset/raw/legitphish.csv")

print(data.head())