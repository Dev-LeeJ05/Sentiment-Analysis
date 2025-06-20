import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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