import torch
import torch.nn.functional as F

class CustomDataCollatorForSequenceClassification:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have 'pad_token_id' attribute defined.")
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths)

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []

        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            token_type_ids = feature["token_type_ids"]
            label = feature["labels"]

            current_length = lengths[i]
            padding_length = max_length - current_length
            
            padded_input_ids = F.pad(input_ids, (0, padding_length), value=self.pad_token_id)
            padded_attention_mask = F.pad(attention_mask, (0, padding_length), value=0)
            padded_token_type_ids = F.pad(token_type_ids, (0, padding_length), value=0)
            
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_token_type_ids.append(padded_token_type_ids)
            
            batch_labels.append(label)

        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
        batch_token_type_ids = torch.stack(batch_token_type_ids, dim=0)
        batch_labels = torch.stack(batch_labels, dim=0)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "token_type_ids": batch_token_type_ids,
            "labels": batch_labels,
        }