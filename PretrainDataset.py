import torch
import random
from torch.utils.data import Dataset

class CustomDataCollatorForMLM:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_tokens_mask_ids = set([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ])

    def __call__(self, examples):
        batch = {}
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        for example in examples:
            input_ids_list.append(example["input_ids"])
            attention_mask_list.append(example["attention_mask"])
            token_type_ids_list.append(example["token_type_ids"])
        
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids_list, dtype=torch.long)

        inputs, labels = self.mask_tokens(input_ids)
        
        batch["input_ids"] = inputs
        batch["attention_mask"] = attention_mask
        batch["token_type_ids"] = token_type_ids
        batch["labels"] = labels
        return batch

    def mask_tokens(self, inputs: torch.Tensor):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        for i in range(labels.shape[0]):
            special_tokens_mask = torch.tensor([
                1 if labels[i, j].item() in self.special_tokens_mask_ids else 0
                for j in range(labels.shape[1])
            ], dtype=torch.bool)
            probability_matrix[i].masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

class TokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]