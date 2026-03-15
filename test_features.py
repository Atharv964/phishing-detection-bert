from utils.preprocessing import load_and_clean_data
from features.feature_loader import get_handcrafted_features

data = load_and_clean_data("dataset/raw/legitphish.csv")

features = get_handcrafted_features(data)

print("Feature shape:", features.shape)
print(features.head())
