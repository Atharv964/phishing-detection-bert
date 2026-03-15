from transformers import BertTokenizer

# load pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_urls(urls, max_length=64):

    encoded = tokenizer(
        urls.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return encoded
