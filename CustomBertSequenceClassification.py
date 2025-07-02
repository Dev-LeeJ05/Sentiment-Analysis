import os
import torch
import torch.nn as nn
from CustomBert import CustomBertForMaskedLM

class CustomBertSequenceClassification(nn.Module):
    def __init__(self, config, model_weight_path, num_labels):
        super().__init__()
        
        temp_mlm_model = CustomBertForMaskedLM(config)

        if os.path.exists(model_weight_path):
            print(f"모델 가중치를 {model_weight_path}에서 불러오는 중..")
            state_dict = torch.load(model_weight_path, map_location='cpu')
            temp_mlm_model.load_state_dict(state_dict)
            print("모델 가중치 로드 완료")
        else:
            raise FileNotFoundError(f'모델 가중치가 {model_weight_path}에 존재하지 않습니다. 학습을 시작할 수 없습니다.')
        
        self.bert = temp_mlm_model
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else config.dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        encoder_output = self.bert.encoder(embedding_output, extended_attention_mask)
        last_hidden_state = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
        pooled_output = self.bert.pooler(last_hidden_state)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        return {"loss": loss, "logits": logits}