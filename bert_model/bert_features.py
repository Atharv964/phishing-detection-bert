import torch
from transformers import BertModel
from bert_model.tokenizer import tokenize_urls

# load pretrained model
model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embeddings(urls, batch_size=64):

    embeddings_list = []

    for i in range(0, len(urls), batch_size):

        batch_urls = urls[i:i+batch_size]

        tokens = tokenize_urls(batch_urls)

        with torch.no_grad():
            outputs = model(**tokens)

        batch_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings_list.append(batch_embeddings)

        print(f"Processed {i+len(batch_urls)} / {len(urls)} URLs")

    embeddings = torch.cat(embeddings_list, dim=0)

    return embeddings
