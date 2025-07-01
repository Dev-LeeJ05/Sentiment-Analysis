import torch
from torch.utils.data import Dataset
import random

class TokenizedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        processed_item = {}
        for k, v in item.items():
            if isinstance(v, (int, float, bool)):
                processed_item[k] = torch.tensor([v], dtype=torch.long)
            elif isinstance(v, list) and not v:
                processed_item[k] = torch.tensor([], dtype=torch.long)
            elif isinstance(v, list):
                processed_item[k] = torch.tensor(v, dtype=torch.long)
            elif isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    processed_item[k] = v.unsqueeze(0)
                else:
                    processed_item[k] = v
            else:
                processed_item[k] = torch.tensor(v, dtype=torch.long)
        return processed_item

import torch
import random

class CustomDataCollatorForMLM:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]

    def __call__(self, examples):

        filtered_examples = [ex for ex in examples if ex['input_ids'].numel() > 0]
        
        if not filtered_examples:
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "token_type_ids": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long),
            }

        max_length = max([len(ex['input_ids']) for ex in filtered_examples])

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []

        for ex in filtered_examples:
            input_ids = ex['input_ids']
            attention_mask = ex['attention_mask']
            token_type_ids = ex['token_type_ids']

            padding_length = max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            token_type_ids = torch.cat([token_type_ids, torch.zeros(padding_length, dtype=torch.long)])

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
        
        input_ids = torch.stack(batch_input_ids)
        attention_mask = torch.stack(batch_attention_mask)
        token_type_ids = torch.stack(batch_token_type_ids)

        input_ids, labels = self.mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

    def mask_tokens(self, inputs: torch.Tensor):
        labels = inputs.clone()
        
        probability_matrix = torch.full_like(labels, self.mlm_probability, dtype=torch.float)
        
        special_tokens_tensor = torch.tensor(self.special_tokens_mask, device=inputs.device)
        is_special_token_mask = torch.isin(inputs, special_tokens_tensor)
        
        probability_matrix.masked_fill_(is_special_token_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full_like(labels, 0.8, dtype=torch.float)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full_like(labels, 0.5, dtype=torch.float)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.get_vocab_size(), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels