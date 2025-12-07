import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class Attention(nn.Module):
    def __init__(self, batch, seq_len, embed_dim, num_head, dropout = 0.0, proj_dropout = 0.0):
        super(Attention, self).__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_embed = embed_dim // num_head
        
        self.dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        self.query = nn.Linear(embed_dim, embed_dim) #last input dim -> new last input dim
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def softmax_func(self, attn, dim):
        attn_max = attn.max(dim=dim, keepdim=True).values #regularization, reasoning not found
        attn_exp = torch.exp(attn - attn_max)
        softmax_attn = attn_exp / attn_exp.sum(dim=dim, keepdim=True)
        return softmax_attn
        
    def forward(self, x, mask=None):
        batch, seq_len, embed_dim = x.shape

        q = self.query(x).reshape(batch, seq_len, self.num_head, self.head_embed).transpose(1,2)
        k = self.key(x).reshape(batch, seq_len, self.num_head, self.head_embed).transpose(1,2)
        v = self.value(x).reshape(batch, seq_len, self.num_head, self.head_embed).transpose(1,2)

        attn = q @ k.transpose(2,3) / (embed_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = torch.masked_fill(input=attn, mask=mask, value=float("-inf"))
        
        attn = self.softmax_func(attn, dim=-1) # if dim=-2 
        attn = self.dropout(attn)
        attn = attn @ v                        # then attn.transpose(-1,-2) @ v
        
        out = attn.transpose(1,2).flatten(2)
        out = self.proj_dropout(out)
        out = self.out_proj(out)
        
        return out

if __name__ == "__main__":
    
    batch = 5
    num_head = 2
    seq_len = 16
    embed_dim = 32
    s = [16,14,11]
    
    input = torch.randn([len(s), max(s), embed_dim])
    
    attn = Attention(batch, seq_len, embed_dim, num_head)
    
    mask_shape = [torch.ones(seq) for seq in s]
    mask = nn.utils.rnn.pad_sequence(mask_shape, batch_first=True, padding_value=0).bool()
    
    out = attn(input, mask)