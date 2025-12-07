# pos_vs_word_probe.py
# Simple experiments to check whether positional encodings remain recoverable after addition.
# Requires: Python 3.8+, PyTorch, numpy, matplotlib, scikit-learn

import torch, math, numpy as np
import torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

torch.manual_seed(0); np.random.seed(0)

# Settings (adjustable)
d = 1024           # embedding dimension
n_tokens = 600   # number of token instances to sample
rank_word = 8    # intrinsic rank of word subspace
rank_pos = 6     # intrinsic rank of pos subspace
seq_len = 64     # number of distinct positions

# === Build low-rank word subspace and word embeddings ===
U = torch.randn(d, rank_word)               # basis for word-subspace (d x r1)
coeffs = torch.randn(n_tokens, rank_word)   # random coefficients
X_word = coeffs @ U.T                       # (n_tokens, d)

# === Build low-rank random positional subspace (alternative: sinusoidal below) ===
V = torch.randn(d, rank_pos)                # basis for pos-subspace (d x r2)
pos_coeffs = torch.randn(seq_len, rank_pos) * 0.6
pos_table = pos_coeffs @ V.T                # (seq_len, d)

# Sample token positions and assemble instance-level pos vectors
positions = np.random.randint(0, seq_len, size=n_tokens)
X_pos = pos_table[positions]                # (n_tokens, d)

# Mixed embeddings (what a transformer sees when we add PE)
X_mixed = X_word + X_pos

# === Principal angles between subspaces ===
# Orthonormalize bases
Uq, _ = torch.linalg.qr(U, mode='reduced')
Vq, _ = torch.linalg.qr(V, mode='reduced')
C = Uq.T @ Vq
sv = torch.linalg.svdvals(C).cpu().numpy()
angles = np.arccos(np.clip(sv, -1.0, 1.0))  # principal angles (radians)

print("Principal singular values (cosines):", np.round(sv, 4))
print("Principal angles (radians):", np.round(angles, 4))

# === Linear probe helper ===
def train_linear_probe(X_in, y, epochs=300, lr=5e-3):
    model = nn.Linear(d, 1)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = X_in.float(); yt = y.view(-1,1).float()
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    return model, loss.item()

# Target: normalized position scalar in [0,1]
y = torch.tensor(positions / (seq_len - 1), dtype=torch.float32)

# Train probes
_, loss_word = train_linear_probe(X_word, y)
_, loss_pos  = train_linear_probe(X_pos, y)
_, loss_mixed= train_linear_probe(X_mixed, y)

print("\nLinear probe MSEs (predict normalized position):")
print(f"word-only:  {loss_word:.6e}")
print(f"pos-only:   {loss_pos:.6e}")
print(f"mixed (x+e):{loss_mixed:.6e}")

# === Linear decoder: recover e from (x+e) ===
dec = nn.Linear(d, d, bias=False)
opt = optim.Adam(dec.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
Xm = X_mixed.float(); Xp = X_pos.float()
for ep in range(500):
    opt.zero_grad()
    loss = loss_fn(dec(Xm), Xp)
    loss.backward()
    opt.step()
recon_loss = loss.item()
varXp = torch.var(Xp)
explained = 1 - recon_loss / (varXp.item() + 1e-12)
print(f"\nReconstruction MSE of PE from (x+e): {recon_loss:.6e}  (variance explained ≈ {explained:.4f})")

# === Sinusoidal PE experiment ===
def sinusoidal_pos_encoding(seq_len, d):
    enc = np.zeros((seq_len, d))
    for pos in range(seq_len):
        for i in range(0, d, 2):
            enc[pos, i] = math.sin(pos / (10000 ** (i / d)))
            if i + 1 < d:
                enc[pos, i+1] = math.cos(pos / (10000 ** (i / d)))
    return torch.tensor(enc, dtype=torch.float32)

pos_sin_table = sinusoidal_pos_encoding(seq_len, d)
X_pos_sin_inst = pos_sin_table[positions]
X_mixed_sin = X_word + X_pos_sin_inst

_, loss_pos_sin = train_linear_probe(X_pos_sin_inst, y)
_, loss_mixed_sin = train_linear_probe(X_mixed_sin, y)
print("\nSinusoidal PE probe MSEs:")
print(f"pos-only:   {loss_pos_sin:.6e}")
print(f"mixed (x+sinpos): {loss_mixed_sin:.6e}")

# === Quick PCA visual (optional) ===
pca = PCA(n_components=2)
proj_word = pca.fit_transform(X_word.numpy())
proj_pos = pca.transform(X_pos.numpy())
proj_mixed = pca.transform(X_mixed.numpy())

plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.scatter(proj_word[:300,0], proj_word[:300,1], s=8); plt.title("word subspace (PCA)")
plt.subplot(1,3,2); plt.scatter(proj_pos[:300,0], proj_pos[:300,1], s=8); plt.title("pos subspace (PCA)")
plt.subplot(1,3,3); plt.scatter(proj_mixed[:300,0], proj_mixed[:300,1], s=8); plt.title("mixed (x+e) (PCA)")
plt.tight_layout(); plt.show()

print("\nDone. Interpretation:")
print("- If 'mixed' probe MSE ~ 'pos-only' MSE, position is still linearly recoverable after addition.")
print("- Low reconstruction MSE for D means a linear map exists to extract e from x+e.")
print("- Principal singular values near 0 => subspaces are orthogonal; near 1 => aligned.")