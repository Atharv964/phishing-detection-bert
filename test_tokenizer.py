from utils.preprocessing import load_and_clean_data
from bert_model.tokenizer import tokenize_urls

data = load_and_clean_data("dataset/raw/legitphish.csv")

urls = data["URL"].head(5)

tokens = tokenize_urls(urls)

print(tokens)
