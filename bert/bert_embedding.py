import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim):
        super(Embeddings, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.lin_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = self.positional_encoding()
    
    def positional_encoding(self):

        """
            - sin(pos(1/ (10000 ** 2i/dim))) for alternating columns 0 to N, step = 2
            - cos(pos(1/ (10000 ** 2i/dim))) for alternating columns 1 to N, step = 2
            - batch x seq x embed_dim will be the input, pos_encod will be 1 x seq x embed_dim
            - 1 x seq x embed_dim where shape has to be each sin and cos: 1 x 64 x 256
            - here seq = row = 0 to 255, embed_dim = col = 0 to 255
            ? do we need an encoding bigger than 64? Why? Why not? Let's keep it 64 for now. Requires grad or not?
        """        
        encodings = torch.zeros(self.seq_len, self.embed_dim, dtype=torch.float32)
        position_idx = torch.arange(0, self.seq_len, dtype=torch.float32).reshape(-1,1)
        i = torch.arange(0, self.embed_dim, step=2, dtype=torch.float32)
        
        encodings[:,0::2] = torch.sin(position_idx/(10000 ** (2*i/self.embed_dim)))
        encodings[:,1::2] = torch.cos(position_idx/(10000 ** (2*i/self.embed_dim)))

        encodings = nn.Parameter(encodings.unsqueeze(0))
        print(encodings.shape, position_idx.shape, i.shape)        
        print(encodings[:, 0::2].shape, encodings.shape)

        return encodings

    def forward(self, input_ids):
        lin = self.lin_embed(input_ids)
        pos = self.pos_embed
        return lin + pos

if __name__ == "__main__":
    
    vocab_size = 30000
    batch = 3
    seq_len = 64
    embed_dim = 256
    
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    model = Embeddings(vocab_size=vocab_size, seq_len=seq_len, embed_dim=embed_dim)
    
    out = model(input_ids)
    print(out.shape)