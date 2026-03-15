from utils.preprocessing import load_and_clean_data
from bert_model.bert_features import get_bert_embeddings

data = load_and_clean_data("dataset/raw/legitphish.csv")

urls = data["URL"].head(5)

embeddings = get_bert_embeddings(urls)

print("Embedding shape:", embeddings.shape)
