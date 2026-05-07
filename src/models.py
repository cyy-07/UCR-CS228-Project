"""
src/models.py  —  All forecasting models.

TENSOR SHAPE CONVENTION:
    x (input)  : [B, L, C]   B=batch, L=seq_len, C=num_features
    y (output) : [B, H]       H=pred_len

Models
------
  PersistenceBaseline  — last observed target value, no parameters
  MLPForecaster        — flatten → MLP → pred_len
  LSTMForecaster       — stacked LSTM → last hidden → pred_len
  DLinear              — moving-avg decomposition + two linear projections
  PatchTST             — channel-independent patch transformer (simplified)
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  Persistence baseline
# ─────────────────────────────────────────────

class PersistenceBaseline(nn.Module):
    """
    Naïve baseline: repeat the last observed Appliances value for all H steps.
    Zero trainable parameters.
    Input : [B, L, C]   (Appliances history is column 0)
    Output: [B, H]
    """
    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, 0]                          # [B]
        return last.unsqueeze(1).expand(-1, self.pred_len)  # [B, H]


# ─────────────────────────────────────────────
#  MLP
# ─────────────────────────────────────────────

class MLPForecaster(nn.Module):
    """
    Flatten (L×C) → stack of Linear+ReLU+Dropout → pred_len.
    Input : [B, L, C]
    Output: [B, H]
    """
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 hidden: list = None, dropout: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = [256, 128]
        dims = [seq_len * n_feat] + hidden + [pred_len]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))   # [B, L*C] → [B, H]


# ─────────────────────────────────────────────
#  LSTM
# ─────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Stacked LSTM encoder; reads full sequence, uses last hidden state.
    Input : [B, L, C]
    Output: [B, H]
    """
    def __init__(self, pred_len: int, n_feat: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)              # [B, L, hidden]
        return self.head(self.drop(out[:, -1]))   # [B, H]


# ─────────────────────────────────────────────
#  DLinear
# ─────────────────────────────────────────────

class DLinear(nn.Module):
    """
    Decomposition-Linear (Zeng et al., AAAI 2023).
    Splits each channel into trend (moving-avg) + seasonal (residual),
    applies two independent linear maps, sums, then projects n_feat→1.
    Input : [B, L, C]
    Output: [B, H]
    """
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.trend_fc    = nn.Linear(seq_len, pred_len)
        self.seasonal_fc = nn.Linear(seq_len, pred_len)
        self.mix         = nn.Linear(n_feat, 1)   # combine channels

    @staticmethod
    def _moving_avg(x: torch.Tensor, k: int) -> torch.Tensor:
        """Symmetric moving average; x: [B, C, L] → [B, C, L]"""
        pad = (k - 1) // 2
        front = x[:, :, :1].expand(-1, -1, pad)
        back  = x[:, :, -1:].expand(-1, -1, k // 2)
        xp = torch.cat([front, x, back], dim=-1)
        return xp.unfold(-1, k, 1).mean(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] → [B, C, L]
        xT = x.permute(0, 2, 1)
        trend    = self._moving_avg(xT, self.kernel)         # [B, C, L]
        seasonal = xT - trend

        t_out = self.trend_fc(trend)       # [B, C, H]
        s_out = self.seasonal_fc(seasonal) # [B, C, H]
        combined = (t_out + s_out).permute(0, 2, 1)  # [B, H, C]
        return self.mix(combined).squeeze(-1)          # [B, H]


# ─────────────────────────────────────────────
#  PatchTST (simplified, channel-independent)
# ─────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    Simplified channel-independent PatchTST (Nie et al., ICLR 2023).
    Each channel is patched independently, encoded by a shared Transformer,
    mean-pooled, then all channels are concatenated and projected to pred_len.
    Input : [B, L, C]
    Output: [B, H]
    """
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, nhead: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.n_feat    = n_feat
        self.d_model   = d_model
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.proj = nn.Linear(patch_len, d_model)
        self.pos  = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout,
            batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(n_feat * d_model, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        # treat all channels as independent "batch" items
        xc = x.permute(0, 2, 1).reshape(B * C, L)           # [B*C, L]
        patches = xc.unfold(-1, self.patch_len, self.stride) # [B*C, P, plen]
        tok = self.proj(patches) + self.pos                  # [B*C, P, d]
        tok = self.enc(tok).mean(1)                          # [B*C, d]
        tok = tok.reshape(B, C * self.d_model)               # [B, C*d]
        return self.head(self.drop(tok))                     # [B, H]


# ─────────────────────────────────────────────
#  Factory helper
# ─────────────────────────────────────────────

def build_model(name: str, seq_len: int, pred_len: int, n_feat: int) -> nn.Module:
    """Instantiate a model by name string."""
    name = name.lower()
    if name == "persistence":
        return PersistenceBaseline(pred_len)
    if name == "mlp":
        return MLPForecaster(seq_len, pred_len, n_feat)
    if name == "lstm":
        return LSTMForecaster(pred_len, n_feat)
    if name == "dlinear":
        return DLinear(seq_len, pred_len, n_feat)
    if name == "patchtst":
        return PatchTST(seq_len, pred_len, n_feat)
    raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    B, L, C, H = 8, 96, 26, 24
    x = torch.randn(B, L, C)

    for name in ["persistence", "mlp", "lstm", "dlinear", "patchtst"]:
        m   = build_model(name, L, H, C)
        out = m(x)
        assert out.shape == (B, H), f"{name}: {out.shape}"
        npar = sum(p.numel() for p in m.parameters())
        print(f"  {name:<14} out={tuple(out.shape)}  params={npar:,}")

    print("\n✓ models OK")
