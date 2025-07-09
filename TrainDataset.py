from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_classification_dataset(data_frame):
    processed_tokens = []
    processed_attentions = []
    processed_token_type_ids = []

    for i in tqdm(range(len(data_frame)), desc="데이터 파싱 중"):
        token_str = data_frame.iloc[i, 0]
        attention_str = data_frame.iloc[i, 3]
        token_type_ids_str = data_frame.iloc[i, 4]

        processed_tokens.append([int(t) for t in token_str.split(" ")])
        processed_attentions.append([int(a) for a in attention_str.split(" ")])
        processed_token_type_ids.append([int(t) for t in token_type_ids_str.split(" ")])

    dataset_dict = {
        "input_ids": processed_tokens,
        "attention_mask": processed_attentions,
        "token_type_ids": processed_token_type_ids,
        "labels": data_frame["감정"].values.tolist()
    }
        
    return dataset_dict
    
def tensor_dataset(dataset_dict):
    input_ids = torch.tensor(dataset_dict["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(dataset_dict["attention_mask"], dtype=torch.long)
    labels = torch.tensor(dataset_dict["labels"], dtype=torch.long)
    tensorDataset = TensorDataset(input_ids, attention_mask, labels)
    return tensorDataset