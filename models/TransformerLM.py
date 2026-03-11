import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):

        L = x.size(1)
        d = x.size(2)
        
        device = x.device
        dtype = x.dtype
        
        t = torch.arange(L, device=device, dtype=dtype)
        j = torch.arange(0, d, step=2, device=device, dtype=dtype)
        f = torch.exp(-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) * j / d)
        
        t = t[:, None]
        f = f[None, :]
        angles = t * f
        
        sins = torch.sin(angles)
        coss = torch.cos(angles)
        
        pe = torch.stack([sins, coss], dim=-1).reshape(L, d)
        
        pe = pe.unsqueeze(0)
        x = x + pe
        return x
        


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=True)
        self.classifier = nn.Linear(d_model, vocab_size)

    def generateCausalMask(self, L):
        mask = torch.tril(torch.ones(L, L))
        mask = (1 - mask) * float('-inf')
        return mask

    def forward(self, x):
    
        embeddings = self.embeddings(x)
        embeddings_w_pos = self.position(embeddings)
        
        L = x.size(1)
        mask = self.generateCausalMask(L).to(x.device)
        output = self.encoder(embeddings_w_pos, mask=mask, is_causal=True)
        
        scores = self.classifier(output)
        
        return scores

