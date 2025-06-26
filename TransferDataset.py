import torch
from torch.utils.data import Dataset
from tqdm import tqdm
def tokenize_texts(texts, tokenizer, max_length):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []

    for text in tqdm(texts):
        encoded_input = tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
            padding=True
        )
        input_ids_list.append(encoded_input["input_ids"])
        attention_mask_list.append(encoded_input["attention_mask"])
        token_type_ids_list.append(encoded_input["token_type_ids"])

    return {
        "input_ids" : input_ids_list,
        "attention_mask" : attention_mask_list,
        "token_type_ids" : attention_mask_list
    }

class CustomSentimentDataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.input_ids = torch.tensor(tokenized_dataset["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(tokenized_dataset["attention_mask"], dtype=torch.long)
        self.token_type_ids = torch.tensor(tokenized_dataset["token_type_ids"], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __gemitem__(self, idx):
        return{
            "input_ids" : self.input_ids[idx],
            "attention_mask" : self.attention_mask[idx],
            "token_type_ids" : self.token_type_ids[idx],
            "labels" : self.labels[idx]
        }
