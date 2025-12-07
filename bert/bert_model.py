import torch
import torch.nn as nn
from dataclasses import dataclass
import bert_tokenizer
import bert_data_collator

@dataclass
class TransformerConfig:
        
    max_seq_len : int = 64
    embed_dim : int = 256
    num_head : int = 4
    vocab_size : int = 30000
    
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    mlp_ratio: int = 4
    
    encoder_depth: int = 6
    learn_pos_embed: bool = False


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.embed_dim = config.embed_dim
        self.requires_grad = config.learn_pos_embed
        self.lin_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = self.positional_encoding()
    
    def positional_encoding(self):

        """
            - sin(pos(1/ (10000 ** 2i/dim))) for alternating columns 0 to N, step = 2
            - cos(pos(1/ (10000 ** 2i/dim))) for alternating columns 1 to N, step = 2
            - batch x seq x embed_dim will be the input, pos_encod will be 1 x seq x embed_dim
            - 1 x seq x embed_dim where shape has to be each sin and cos: 1 x 64 x 256
            - here seq = row = 0 to 255, embed_dim = col = 0 to 255
            ? do we need an encoding bigger than 64? Why? Why not? Let's keep it 64 for now.
        """        
        encodings = torch.zeros(self.max_seq_len, self.embed_dim, dtype=torch.float32)
        position_idx = torch.arange(0, self.max_seq_len, dtype=torch.float32).reshape(-1,1)
        i = torch.arange(0, self.embed_dim, step=2, dtype=torch.float32)
        
        encodings[:,0::2] = torch.sin(position_idx/(10000 ** (2*i/self.embed_dim)))
        encodings[:,1::2] = torch.cos(position_idx/(10000 ** (2*i/self.embed_dim)))

        encodings = nn.Parameter(encodings.unsqueeze(0), requires_grad=self.requires_grad)

        return encodings

    def forward(self, input_ids):
        lin = self.lin_embed(input_ids)
        pos = self.pos_embed
        return lin + pos


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        
        assert config.embed_dim % config.num_head == 0, "Double check embed_dim per attn_head"
        
        self.embed_dim = config.embed_dim
        self.num_head = config.num_head
        self.head_embed = config.embed_dim // config.num_head
        
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        self.query = nn.Linear(self.embed_dim, self.embed_dim) #last input dim -> new last input dim
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def softmax_func(self, attn, dim):
        attn_max = attn.max(dim=dim, keepdim=True).values #regularization, reasoning not found
        attn_exp = torch.exp(attn - attn_max)
        softmax_attn = attn_exp / attn_exp.sum(dim=dim, keepdim=True)
        return softmax_attn
        
    def forward(self, x, mask=None):
        batch, max_seq_len, embed_dim = x.shape

        q = self.query(x).reshape(batch, max_seq_len, self.num_head, self.head_embed).transpose(1,2)
        k = self.key(x).reshape(batch, max_seq_len, self.num_head, self.head_embed).transpose(1,2)
        v = self.value(x).reshape(batch, max_seq_len, self.num_head, self.head_embed).transpose(1,2)

        attn = q @ k.transpose(2,3) / (embed_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = torch.masked_fill(input=attn, mask=mask, value=float("-inf"))
        
        attn = self.softmax_func(attn, dim=-1) # if dim=-2 
        attn = self.dropout(attn)
        attn = attn @ v                        # then attn.transpose(-1,-2) @ v
        
        out = attn.transpose(1,2).flatten(2)
        out = self.attention_dropout(out)
        out = self.out_proj(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()

        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.hidden_embed = self.embed_dim * self.mlp_ratio
        
        self.hidden_layer = nn.Linear(self.embed_dim, self.hidden_embed)
        self.activation = nn.GELU()
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        
        self.output_layer = nn.Linear(self.hidden_embed, self.embed_dim)
        self.output_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.hidden_dropout(x)

        x = self.output_layer(x)
        x = self.output_dropout(x)

        return x
    
class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.attn = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.embed_dim)

        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x = self.layer_norm(x)

        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)

        return x


class MLMHead(nn.Module):
    def __init__(self, config):
        super(MLMHead, self).__init__()
        # Dense , GELU , Dropout , Linear Projection to logits
        self.dense_mlm = nn.Linear(config.embed_dim, config.embed_dim)
        self.activation_mlm = nn.GELU()
        self.layer_norm_mlm = nn.LayerNorm(config.embed_dim)

        self.decoder = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        x = self.dense_mlm(x)
        x = self.activation_mlm(x)
        x = self.layer_norm_mlm(x)
        logits = self.decoder(x)
        return logits

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.embedding = Embeddings(config)
        
        self.encoder = nn.ModuleList(
            [Encoder(config) for _ in range(config.encoder_depth)]
        )
        
        self.mlm_head = MLMHead(config)
        self.apply(_init_weights_)

        self.mlm_head.decoder.weight = self.embedding.lin_embed.weight

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        for layer in self.encoder:
            embed = layer(embed)
        
        pred = self.mlm_head(embed)

        return pred
    
    @torch.no_grad()
    def inference(self, input_ids, mask_token_id, top_k):
        x = self.embedding(input_ids)
        for layer in self.encoder:
            x = layer(x)

        logits = self.mlm_head(x)   # (batch, seq, vocab)
        probs = logits.softmax(dim=-1)

        # Mask positions
        mask_positions = (input_ids == mask_token_id)  # (batch, seq)

        results = []

        for b in range(input_ids.size(0)):
            sample_predictions = []
            for pos in torch.where(mask_positions[b])[0]:
                topk = torch.topk(probs[b, pos], k=top_k)
                sample_predictions.append({
                    "position": pos.item(),
                    "topk_ids": topk.indices.tolist(),
                    "topk_probs": topk.values.tolist()
                })
            results.append(sample_predictions)

        return results

def _init_weights_(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_() #zero out any padding weights
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0) #start with identity matrix
        module.bias.data.zero_()

if __name__ == "__main__":
    
    path_to_vocab = "bert_tokenizer.json"
    
    config = TransformerConfig()
    model = Transformer(config)
    
    tok_wrapper = bert_tokenizer.BERTWPWrapper(path_to_vocab=path_to_vocab, truncate=True, max_length=64)
    tokenizer = tok_wrapper.tokenizer

    data_collator  = bert_data_collator.DataCollator(max_length=config.max_seq_len, tokenizer=tokenizer)
    
    text = "The best religion is [MASK]."
    input_ids = tokenizer.encode(text)
    input_ids = input_ids.ids
    print(input_ids)
    input_ids = data_collator.padding(input_ids)

    model.eval()
    preds = model.inference(
        input_ids=torch.tensor([input_ids]),
        mask_token_id=tokenizer.token_to_id("[MASK]"),
        top_k=5
    )
    
    print(preds)
    print(tokenizer.decode(preds[0][0]["topk_ids"]))