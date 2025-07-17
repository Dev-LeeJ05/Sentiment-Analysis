import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p=0.4, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] 
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_dim, output_dim, n_layers,n_heads,dropout_p=0.4,max_len=128,pad_token_id=0):
        super().__init__()
        self.d_model = embedding_dim
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size,self.d_model,padding_idx=pad_token_id,)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_p, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_p,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=n_layers)
        self.fc_out = nn.Linear(self.d_model,output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = self.pos_encoder(embedded)
        padding_mask = (attention_mask == 0)
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        cls_output = transformer_output[:,0,:]
        x = self.dropout(cls_output)
        
        output = self.fc_out(x)
        return output


class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,dropout_p=0.1,bidirectional=False):
        super().__init__()
        self.factor = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,bidirectional=bidirectional,dropout=dropout_p if n_layers > 1 else 0)
        self.f = nn.Linear(hidden_dim * self.factor,output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,x,attention_mask):
        embedded = self.dropout(self.embedding(x))
        lengths = attention_mask.sum(dim=1)
        embedded = embedded.permute(1,0,2)

        packed_embedded = pack_padded_sequence(embedded,lengths.cpu(),enforce_sorted=False)

        x, (h, c) = self.lstm(packed_embedded)
        
        if self.lstm.bidirectional:
            h = self.dropout(torch.cat((h[-2,:,:],h[-1,:,:]),dim=1))
        else:
            h = self.dropout(h[-1,:,:])
        x = self.f(h)
        return x

class SentimentClassifier(nn.Module):
    def __init__(self, bert_encoder, num_labels):
        super(SentimentClassifier,self).__init__()
        self.bert = bert_encoder
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        pooled_output = outputs[0][:, 0]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits