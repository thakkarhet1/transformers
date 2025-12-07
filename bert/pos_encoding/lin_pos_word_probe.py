import torch
import math

# -------------------------
# Settings
# -------------------------
batch_size = 20
seq_len = 50
dim = 64  # high-dimensional embeddings

# -------------------------
# 1. Random word embeddings
# -------------------------
word_embed = torch.randn(batch_size, seq_len, dim)

# -------------------------
# 2. Sinusoidal positional embeddings
# -------------------------
def sinusoidal_pos_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pos_embed = sinusoidal_pos_encoding(seq_len, dim).unsqueeze(0)  # shape: (1, seq_len, dim)

# -------------------------
# 3. Sum embeddings
# -------------------------
x = word_embed + pos_embed  # shape: (batch, seq, dim)

# -------------------------
# 4. "Projections" to recover each component
# -------------------------
# For demonstration, we pretend we somehow know the directions (identity here)
# In practice, a learned linear layer W would learn these projections
proj_word = torch.eye(dim)  # identity projection along word embedding "directions"
proj_pos = torch.eye(dim)   # identity projection along positional embedding "directions"

# Approximate recovery
recovered_word = x @ proj_word  # batch @ seq @ dim
recovered_pos = x @ proj_pos - recovered_word  # naive subtraction to mimic disentangling

# -------------------------
# 5. Check recovery
# -------------------------
word_error = (recovered_word - word_embed).abs().mean()
pos_error = (recovered_pos - pos_embed).abs().mean()

print("Mean absolute error (word):", word_error.item())
print("Mean absolute error (pos):", pos_error.item())