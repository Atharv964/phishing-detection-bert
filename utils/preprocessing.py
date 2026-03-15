import pandas as pd

def load_and_clean_data(path):

    data = pd.read_csv(path)

    print("Original dataset shape:", data.shape)

    # remove missing values
    data = data.dropna()

    # remove duplicate URLs
    data = data.drop_duplicates(subset="URL")

    # convert label to integer
    data["ClassLabel"] = data["ClassLabel"].astype(int)

    # reset index
    data = data.reset_index(drop=True)

    print("Cleaned dataset shape:", data.shape)

    return data
