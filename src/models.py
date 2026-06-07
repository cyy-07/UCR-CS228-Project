"""
src/models.py — All forecasting models.

TENSOR SHAPE CONVENTION:
    x (input)  : [B, L, C]   B=batch, L=seq_len, C=num_features
    y (output) : [B, H]      H=pred_len

Models
------
  PersistenceBaseline   — last observed target value, no parameters
  MLPForecaster         — flatten → MLP → pred_len
  LSTMForecaster        — stacked LSTM → last hidden → pred_len
  DLinear               — moving-avg decomposition + two linear projections
  PatchTST              — channel-independent patch transformer (simplified)
  ITransformer          — Inverted-Transformer (Liu et al., ICLR 2024)

All deep models accept `use_residual=True` (default), wrapping their output as
    output = persistence_baseline + learned_residual
which provides a strong inductive bias for forecasting tasks.
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  Persistence baseline
# ─────────────────────────────────────────────

class PersistenceBaseline(nn.Module):
    """Repeat the last Appliances value for all H steps. Zero parameters."""
    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, 0:1].expand(-1, self.pred_len)        # [B, H]


# ─────────────────────────────────────────────
#  Seasonal-naive baseline  (long-horizon weekly task)
# ─────────────────────────────────────────────

class SeasonalNaiveBaseline(nn.Module):
    """
    Predict each future step by copying the value exactly one period ago
    ("next Monday 9am = last Monday 9am"). Zero parameters.

    For the hourly weekly task (period=168 h) with seq_len=pred_len=168 this
    is simply "replay last week verbatim" — a deliberately strong, intuitive
    baseline that any useful model must beat. Requires seq_len >= period.
    """
    def __init__(self, pred_len: int, period: int = 168):
        super().__init__()
        self.pred_len = pred_len
        self.period   = period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        # forecast step k (0-indexed) copies the target value `period` steps
        # before t+k+1, i.e. input index  L - period + (k % period).
        idx = [L - self.period + (k % self.period) for k in range(self.pred_len)]
        return x[:, idx, 0]                                    # [B, H]


# ─────────────────────────────────────────────
#  Helper: persistence anchor
# ─────────────────────────────────────────────

def _persistence_anchor(x: torch.Tensor, pred_len: int) -> torch.Tensor:
    """[B, L, C] → [B, H] by repeating the last value of channel 0."""
    return x[:, -1, 0:1].expand(-1, pred_len)


# ─────────────────────────────────────────────
#  MLP
# ─────────────────────────────────────────────

class MLPForecaster(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 hidden=None, dropout=0.1, use_residual: bool = True):
        super().__init__()
        self.pred_len = pred_len
        self.use_residual = use_residual
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
        out = self.net(x.flatten(1))
        if self.use_residual:
            out = out + _persistence_anchor(x, self.pred_len)
        return out


# ─────────────────────────────────────────────
#  LSTM
# ─────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, pred_len: int, n_feat: int,
                 hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        self.pred_len = pred_len
        self.use_residual = use_residual
        self.lstm = nn.LSTM(n_feat, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = self.head(self.drop(out[:, -1]))
        if self.use_residual:
            h = h + _persistence_anchor(x, self.pred_len)
        return h


# ─────────────────────────────────────────────
#  DLinear
# ─────────────────────────────────────────────

class DLinear(nn.Module):
    """Decomposition-Linear (Zeng et al., AAAI 2023)."""
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 kernel: int = 25, use_residual: bool = True):
        super().__init__()
        self.pred_len    = pred_len
        self.use_residual = use_residual
        self.kernel      = kernel
        self.trend_fc    = nn.Linear(seq_len, pred_len)
        self.seasonal_fc = nn.Linear(seq_len, pred_len)
        self.mix         = nn.Linear(n_feat, 1)

    @staticmethod
    def _moving_avg(x: torch.Tensor, k: int) -> torch.Tensor:
        pad = (k - 1) // 2
        front = x[:, :, :1].expand(-1, -1, pad)
        back  = x[:, :, -1:].expand(-1, -1, k // 2)
        xp = torch.cat([front, x, back], dim=-1)
        return xp.unfold(-1, k, 1).mean(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xT = x.permute(0, 2, 1)                         # [B, C, L]
        trend    = self._moving_avg(xT, self.kernel)
        seasonal = xT - trend
        t_out = self.trend_fc(trend)                    # [B, C, H]
        s_out = self.seasonal_fc(seasonal)
        combined = (t_out + s_out).permute(0, 2, 1)     # [B, H, C]
        out = self.mix(combined).squeeze(-1)            # [B, H]
        if self.use_residual:
            out = out + _persistence_anchor(x, self.pred_len)
        return out


# ─────────────────────────────────────────────
#  PatchTST (simplified, channel-independent)
# ─────────────────────────────────────────────

class PatchTST(nn.Module):
    """Simplified channel-independent PatchTST (Nie et al., ICLR 2023)."""
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, nhead: int = 4,
                 n_layers: int = 2, dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        self.pred_len  = pred_len
        self.use_residual = use_residual
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
        xc = x.permute(0, 2, 1).reshape(B * C, L)
        patches = xc.unfold(-1, self.patch_len, self.stride)
        tok = self.proj(patches) + self.pos
        tok = self.enc(tok).mean(1)
        tok = tok.reshape(B, C * self.d_model)
        out = self.head(self.drop(tok))
        if self.use_residual:
            out = out + _persistence_anchor(x, self.pred_len)
        return out


# ─────────────────────────────────────────────
#  iTransformer (Liu et al., ICLR 2024)
# ─────────────────────────────────────────────

class ITransformer(nn.Module):
    """
    Inverted Transformer: each variate (channel) becomes a token, and the
    Transformer attends across variates rather than across time. This often
    matches/exceeds PatchTST on multivariate forecasting benchmarks.

    Pipeline
    --------
    1. embed each channel's full L-step series into a d_model-vector  (C tokens)
    2. Transformer encoder mixes the C tokens (cross-variate attention)
    3. project the target channel's token to pred_len
    """
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 d_model: int = 128, nhead: int = 4,
                 n_layers: int = 2, dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        self.pred_len = pred_len
        self.n_feat   = n_feat
        self.d_model  = d_model
        self.use_residual = use_residual

        self.embed = nn.Linear(seq_len, d_model)             # per-channel
        enc_layer  = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout,
            batch_first=True, norm_first=True)
        self.enc   = nn.TransformerEncoder(enc_layer, n_layers)
        self.head  = nn.Linear(d_model, pred_len)            # only on target token
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] → [B, C, L] → embed each channel as one token of d_model
        xT  = x.permute(0, 2, 1)                              # [B, C, L]
        tok = self.embed(xT)                                  # [B, C, d_model]
        tok = self.enc(tok)                                   # [B, C, d_model]
        # use the TARGET channel (index 0) token to produce H-step output
        out = self.head(self.drop(tok[:, 0]))                 # [B, H]
        if self.use_residual:
            out = out + _persistence_anchor(x, self.pred_len)
        return out


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  DPMixer  (our proposed architecture, Workstream G)
# ─────────────────────────────────────────────

class _DLinearTarget(nn.Module):
    """Series-decomposition linear projection of the TARGET channel only."""
    def __init__(self, seq_len: int, pred_len: int, kernel: int = 25):
        super().__init__()
        self.kernel      = kernel
        self.trend_fc    = nn.Linear(seq_len, pred_len)
        self.seasonal_fc = nn.Linear(seq_len, pred_len)

    @staticmethod
    def _moving_avg(x: torch.Tensor, k: int) -> torch.Tensor:
        pad   = (k - 1) // 2
        front = x[:, :, :1].expand(-1, -1, pad)
        back  = x[:, :, -1:].expand(-1, -1, k // 2)
        xp    = torch.cat([front, x, back], dim=-1)
        return xp.unfold(-1, k, 1).mean(-1)

    def forward(self, target_hist: torch.Tensor) -> torch.Tensor:
        """target_hist: [B, L] → [B, H]."""
        x = target_hist.unsqueeze(1)              # [B, 1, L]
        trend    = self._moving_avg(x, self.kernel)
        seasonal = x - trend
        return (self.trend_fc(trend) + self.seasonal_fc(seasonal)).squeeze(1)


class _MixerBlock(nn.Module):
    """Tiny MLP-Mixer block: independent mix along one axis."""
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden)
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [..., dim] — mix on the last axis
        y = self.norm(x)
        y = self.fc2(self.drop(torch.relu(self.fc1(y))))
        return x + y


class DPMixer(nn.Module):
    """
    Decomposition-Persistence Mixer with Learned Branch Gating.

    Branches (additive):
      1. Persistence anchor:      x[:, -1, 0:1] expanded over H
                                  (cf. NLinear, Zeng et al. AAAI 2023)
      2. Target decomposition:    DLinear-style trend+seasonal on target only
                                  (cf. DLinear, Zeng et al. AAAI 2023)
      3. Cross-variate channel mixer (MLP-Mixer over the C dimension at last
         timestep) → linear → H
                                  (cf. TSMixer, Chen et al. TMLR 2023)
      4. Cross-time time mixer    (MLP-Mixer over the L dimension on the target
         channel only) → linear → H
                                  (cf. TSMixer, Chen et al. TMLR 2023)

    Output = α₁·B₁ + α₂·B₂ + α₃·B₃ + α₄·B₄   (learned scalar gates αᵢ)

    Our contribution: every branch is borrowed from a well-known prior
    architecture — but the LEARNED PER-BRANCH GATES are new. The four gates
    are reported after training as a small interpretability tool that tells
    us which inductive bias mattered most on this dataset.
    """
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 d_channel: int = 64, d_time: int = 128,
                 dropout: float = 0.1, use_decomp: bool = True,
                 use_channel: bool = True, use_time: bool = True,
                 use_residual: bool = True, use_gating: bool = True):
        super().__init__()
        self.pred_len     = pred_len
        self.use_residual = use_residual
        self.use_decomp   = use_decomp
        self.use_channel  = use_channel
        self.use_time     = use_time
        self.use_gating   = use_gating

        if use_decomp:
            self.decomp = _DLinearTarget(seq_len, pred_len)

        if use_channel:
            # take the last timestep across C channels → mix → project to H
            self.ch_norm = nn.LayerNorm(n_feat)
            self.ch_mix1 = _MixerBlock(n_feat, n_feat * 2, dropout)
            self.ch_mix2 = _MixerBlock(n_feat, n_feat * 2, dropout)
            self.ch_head = nn.Linear(n_feat, pred_len)

        if use_time:
            self.tm_mix1 = _MixerBlock(seq_len, d_time, dropout)
            self.tm_mix2 = _MixerBlock(seq_len, d_time, dropout)
            self.tm_head = nn.Linear(seq_len, pred_len)

        # Four scalar gates initialized to 1.0 (i.e., default = unweighted sum).
        # No positivity constraint — letting them go negative is informative.
        if use_gating:
            self.gate = nn.Parameter(torch.ones(4))

    def _zeros(self, B, device):
        return torch.zeros(B, self.pred_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Branch 1: persistence anchor
        b1 = x[:, -1, 0:1].expand(-1, self.pred_len) if self.use_residual \
             else self._zeros(B, x.device)

        # Branch 2: target decomposition
        b2 = self.decomp(x[:, :, 0]) if self.use_decomp \
             else self._zeros(B, x.device)

        # Branch 3: cross-variate channel mixer (last timestep)
        if self.use_channel:
            ch = self.ch_norm(x[:, -1, :])
            ch = self.ch_mix2(self.ch_mix1(ch))
            b3 = self.ch_head(ch)
        else:
            b3 = self._zeros(B, x.device)

        # Branch 4: cross-time time mixer (target channel only)
        if self.use_time:
            tm = self.tm_mix2(self.tm_mix1(x[:, :, 0]))
            b4 = self.tm_head(tm)
        else:
            b4 = self._zeros(B, x.device)

        if self.use_gating:
            g = self.gate
            return g[0] * b1 + g[1] * b2 + g[2] * b3 + g[3] * b4
        return b1 + b2 + b3 + b4

    def report_gates(self) -> dict:
        """For interpretability — print learned branch weights."""
        if not self.use_gating:
            return {}
        names = ["persistence", "decomp", "channel", "time"]
        return {n: round(float(self.gate[i]), 3) for i, n in enumerate(names)}


# ─────────────────────────────────────────────
#  TC-DPMixer  —  Time-Conditioned gating  (Idea A)
# ─────────────────────────────────────────────

class TCDPMixer(DPMixer):
    """
    Time-Conditioned DPMixer.  The four branch gates become INPUT-DEPENDENT,
    produced by a tiny MLP that consumes the time-of-day / weekday features
    at the LAST input timestep:

        αᵢ(t) = sigmoid(MLP(time_feats(t))) · 2     # scaled to [0, 2]

    Related work:
      • MoLE (Ni et al., AISTATS 2024) — timestamp-conditioned routing over
        DLinear experts. We extend this idea to a HETEROGENEOUS additive
        mixer (persistence + decomp + channel-mix + time-mix instead of
        four linear experts), and condition on FORECAST-step time features
        rather than the input start timestamp.
      • TFT (Lim et al., 2021) — variable-selection gates conditioned on
        observed/known covariates.

    The model expects the LAST `time_feat_dim` channels of x to be time
    features (sin/cos hour, sin/cos weekday, is_weekend, [nsm]). This
    matches data_utils.py's `add_time_features=True` layout.
    """

    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 time_feat_dim: int = 6, gate_hidden: int = 16,
                 **dpmixer_kwargs):
        # Force gating off in parent — we override with input-dependent gates.
        dpmixer_kwargs["use_gating"] = False
        super().__init__(seq_len, pred_len, n_feat, **dpmixer_kwargs)
        self.time_feat_dim = time_feat_dim
        assert n_feat >= time_feat_dim, (
            f"TCDPMixer needs at least {time_feat_dim} time features; "
            f"got n_feat={n_feat}. Run with data_utils Config("
            f"add_time_features=True).")
        self.use_tc_gating = True   # distinguishes from base DPMixer flag

        # Tiny MLP: time_feats → 4 gates
        self.gate_net = nn.Sequential(
            nn.Linear(time_feat_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 4),
        )
        # Initialize last layer to zero so initial gates ≈ sigmoid(0)·2 = 1.0
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def _compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample gates from time features at the last input step."""
        tf = x[:, -1, -self.time_feat_dim:]              # [B, time_feat_dim]
        return torch.sigmoid(self.gate_net(tf)) * 2.0    # [B, 4]  ∈ [0, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # Re-compute branches (cannot use super().forward because we need
        # access to b1..b4 individually for per-sample gating).
        b1 = x[:, -1, 0:1].expand(-1, self.pred_len) if self.use_residual \
             else self._zeros(B, x.device)
        b2 = self.decomp(x[:, :, 0]) if self.use_decomp \
             else self._zeros(B, x.device)
        if self.use_channel:
            ch = self.ch_norm(x[:, -1, :])
            ch = self.ch_mix2(self.ch_mix1(ch))
            b3 = self.ch_head(ch)
        else:
            b3 = self._zeros(B, x.device)
        if self.use_time:
            tm = self.tm_mix2(self.tm_mix1(x[:, :, 0]))
            b4 = self.tm_head(tm)
        else:
            b4 = self._zeros(B, x.device)

        g = self._compute_gates(x)                        # [B, 4]
        return (g[:, 0:1] * b1 + g[:, 1:2] * b2 +
                g[:, 2:3] * b3 + g[:, 3:4] * b4)

    @torch.no_grad()
    def hourly_gate_schedule(self, weekday: int = 2) -> torch.Tensor:
        """
        Probe the learned 24-h schedule of branch trust on a chosen weekday.
        Returns tensor of shape [24, 4] with gate values at hours 0..23.
        Useful for the headline visualization in the report.
        """
        device = next(self.parameters()).device
        hours = torch.arange(24, device=device).float()
        h_sin = torch.sin(2 * math.pi * hours / 24.0)
        h_cos = torch.cos(2 * math.pi * hours / 24.0)
        w_sin = torch.full_like(hours, math.sin(2 * math.pi * weekday / 7.0))
        w_cos = torch.full_like(hours, math.cos(2 * math.pi * weekday / 7.0))
        is_wk = torch.zeros_like(hours)
        nsm   = hours * 3600.0
        # Build the time-feature row matching data_utils.TIME order:
        #   ["hour_sin","hour_cos","weekday_sin","weekday_cos","is_weekend","nsm"]
        feats = torch.stack([h_sin, h_cos, w_sin, w_cos, is_wk, nsm], -1)
        feats = feats[:, : self.time_feat_dim]            # [24, time_feat_dim]
        # Note: training data is normalized; we approximate by passing raw
        # features here — gate VALUES are interpreted qualitatively only.
        return torch.sigmoid(self.gate_net(feats)) * 2.0  # [24, 4]


# ─────────────────────────────────────────────
#  TC-Gated DLinear  —  transferability probe for the TC mechanism
# ─────────────────────────────────────────────

class TCGatedDLinear(nn.Module):
    """DLinear whose trend & seasonal branches are combined by GATES.

    This exists to test whether the time-conditioned gating mechanism that
    helps DPMixer also helps a DIFFERENT additive model — here a 2-branch
    (trend + seasonal) DLinear rather than DPMixer's 4 branches. If TC gating
    improves both, the contribution is a reusable MECHANISM, not a one-model
    trick.

        use_tc_gating=True   → 2 per-sample gates from time features (the TC
                               mechanism)
        use_tc_gating=False  → 2 static learned scalar gates (control)
    """
    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 kernel: int = 25, time_feat_dim: int = 6,
                 gate_hidden: int = 16, use_residual: bool = True,
                 use_tc_gating: bool = True):
        super().__init__()
        self.pred_len      = pred_len
        self.kernel        = kernel
        self.use_residual  = use_residual
        self.use_tc_gating = use_tc_gating
        self.time_feat_dim = time_feat_dim
        self.trend_fc      = nn.Linear(seq_len, pred_len)
        self.seasonal_fc   = nn.Linear(seq_len, pred_len)
        self.mix           = nn.Linear(n_feat, 1)
        if use_tc_gating:
            assert n_feat >= time_feat_dim, (
                f"TCGatedDLinear needs >= {time_feat_dim} time features; "
                f"got n_feat={n_feat}. Use add_time_features=True.")
            self.gate_net = nn.Sequential(
                nn.Linear(time_feat_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 2),
            )
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)
        else:
            self.gate = nn.Parameter(torch.ones(2))

    @staticmethod
    def _moving_avg(x: torch.Tensor, k: int) -> torch.Tensor:
        pad   = (k - 1) // 2
        front = x[:, :, :1].expand(-1, -1, pad)
        back  = x[:, :, -1:].expand(-1, -1, k // 2)
        xp    = torch.cat([front, x, back], dim=-1)
        return xp.unfold(-1, k, 1).mean(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xT       = x.permute(0, 2, 1)                   # [B, C, L]
        trend    = self._moving_avg(xT, self.kernel)
        seasonal = xT - trend
        t_out = self.mix(self.trend_fc(trend).permute(0, 2, 1)).squeeze(-1)
        s_out = self.mix(self.seasonal_fc(seasonal).permute(0, 2, 1)).squeeze(-1)
        if self.use_tc_gating:
            tf = x[:, -1, -self.time_feat_dim:]
            g  = torch.sigmoid(self.gate_net(tf)) * 2.0  # [B, 2] ∈ [0, 2]
            out = g[:, 0:1] * t_out + g[:, 1:2] * s_out
        else:
            out = self.gate[0] * t_out + self.gate[1] * s_out
        if self.use_residual:
            out = out + _persistence_anchor(x, self.pred_len)
        return out


# ─────────────────────────────────────────────
#  SS-DPMixer  —  Spike-aware two-stream  (Idea C, NILM-inspired)
# ─────────────────────────────────────────────

class SSDPMixer(nn.Module):
    """
    Spike-aware two-stream DPMixer.

    Architecture
    ------------
    BASE stream (smooth load): persistence anchor + DLinear-style decomposition
    SPIKE stream (high-freq events): channel mixer + time mixer →
       (a) spike_classifier  — per-step logit p(spike)
       (b) spike_magnitude   — per-step magnitude m
    Final output  ŷ_t = base(t) + σ(p(t)) · m(t)

    Related work
    ------------
      • Zhang et al., AAAI 2018 — Sequence-to-Point NILM (classify + regress
        appliance activations; arXiv:1612.09106). Our setup borrows the
        classify-and-magnitude head pattern but applies it to FORECASTING
        rather than disaggregation.
      • EPF spike-decomposition (Lago/Marcjasz line) — classical statistical
        precedent for two-stream spike modelling on electricity series.

    Honest novelty claim: applying NILM-style auxiliary heads to a future
    spike indicator + magnitude on top of a smooth base-load forecaster on
    the UCI Appliances dataset.
    """

    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 d_channel: int = 64, d_time: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len

        # ── Base stream: persistence + DLinear-on-target ─────────
        self.base_decomp = _DLinearTarget(seq_len, pred_len)

        # ── Spike stream: channel + time mixers, shared trunk ─
        self.ch_norm = nn.LayerNorm(n_feat)
        self.ch_mix1 = _MixerBlock(n_feat, n_feat * 2, dropout)
        self.ch_mix2 = _MixerBlock(n_feat, n_feat * 2, dropout)
        self.ch_proj = nn.Linear(n_feat,  pred_len)

        self.tm_mix1 = _MixerBlock(seq_len, d_time, dropout)
        self.tm_mix2 = _MixerBlock(seq_len, d_time, dropout)
        self.tm_proj = nn.Linear(seq_len, pred_len)

        self.spike_clf  = nn.Linear(pred_len * 2, pred_len)  # classifier head
        self.spike_mag  = nn.Linear(pred_len * 2, pred_len)  # magnitude head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # base stream
        anchor   = x[:, -1, 0:1].expand(-1, self.pred_len)
        base     = anchor + self.base_decomp(x[:, :, 0])      # [B, H]

        # spike stream
        ch = self.ch_norm(x[:, -1, :])
        ch = self.ch_mix2(self.ch_mix1(ch))
        ch_out = self.ch_proj(ch)                              # [B, H]
        tm = self.tm_mix2(self.tm_mix1(x[:, :, 0]))
        tm_out = self.tm_proj(tm)                              # [B, H]
        feat = torch.cat([ch_out, tm_out], dim=-1)             # [B, 2H]

        spike_logit = self.spike_clf(feat)                     # [B, H]
        spike_mag   = self.spike_mag(feat)                     # [B, H]
        spike_prob  = torch.sigmoid(spike_logit)

        return base + spike_prob * spike_mag                   # [B, H]


# ─────────────────────────────────────────────
#  TC-SS-DPMixer  —  combined A + C  (our headline contribution)
# ─────────────────────────────────────────────

class TCSSDPMixer(SSDPMixer):
    """
    Time-Conditioned, Spike-Aware DPMixer.

    Same as SSDPMixer but with a small MLP that produces TWO per-sample
    gates conditioned on the input's last-timestep time features:
        ŷ = α_base(t)·base + α_spike(t)·(σ(spike_logit) · spike_mag)

    Story: the model learns to TRUST the base stream overnight and the
    spike stream around meal-time peaks — exactly the kind of human
    behaviour the data captures.
    """

    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 time_feat_dim: int = 6, gate_hidden: int = 16,
                 **kwargs):
        super().__init__(seq_len, pred_len, n_feat, **kwargs)
        self.time_feat_dim = time_feat_dim
        assert n_feat >= time_feat_dim, (
            f"TCSSDPMixer needs at least {time_feat_dim} time features.")
        self.gate_net = nn.Sequential(
            nn.Linear(time_feat_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 2),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        anchor = x[:, -1, 0:1].expand(-1, self.pred_len)
        base   = anchor + self.base_decomp(x[:, :, 0])
        ch = self.ch_norm(x[:, -1, :])
        ch = self.ch_mix2(self.ch_mix1(ch))
        ch_out = self.ch_proj(ch)
        tm = self.tm_mix2(self.tm_mix1(x[:, :, 0]))
        tm_out = self.tm_proj(tm)
        feat = torch.cat([ch_out, tm_out], dim=-1)
        spike_prob = torch.sigmoid(self.spike_clf(feat))
        spike_mag  = self.spike_mag(feat)
        spike_out  = spike_prob * spike_mag

        tf = x[:, -1, -self.time_feat_dim:]
        g  = torch.sigmoid(self.gate_net(tf)) * 2.0           # [B, 2] ∈ [0,2]
        return g[:, 0:1] * base + g[:, 1:2] * spike_out


# ─────────────────────────────────────────────
#  TC-Spike Classifier  —  binary 'spike in next H steps'  (Task #4)
# ─────────────────────────────────────────────

class TCSpikeClassifier(TCDPMixer):
    """Time-conditioned spike CLASSIFIER (binary per-step).

    Architecture is identical to TCDPMixer (three TC-gated branches: target
    decomp + cross-variate channel mixer + cross-time time mixer), but:
      • The persistence anchor is FORCED OFF.  Anchoring a binary classifier
        on a thresholded raw value is unprincipled (it would just hard-copy
        the spikiness of the last input step into every future step).
      • `forward()` returns H LOGITS, not Wh.  Use `predict_proba()` to get
        per-step spike probabilities ∈ [0, 1].
      • Trained with `BCEWithLogitsLoss` (numerically stable; computes the
        sigmoid internally).

    The 3 surviving branches still output `[B, H]` per branch — those values
    play the role of un-normalized logits when summed under the TC gates.
    BCE will adapt their scale during training; no extra projection needed.
    """

    def __init__(self, seq_len: int, pred_len: int, n_feat: int,
                 time_feat_dim: int = 6, gate_hidden: int = 16,
                 **dpmixer_kwargs):
        # Persistence anchor disabled (see docstring rationale).
        dpmixer_kwargs["use_residual"] = False
        super().__init__(seq_len, pred_len, n_feat,
                         time_feat_dim=time_feat_dim,
                         gate_hidden=gate_hidden, **dpmixer_kwargs)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-step spike probability tensor of shape [B, H] ∈ [0, 1]."""
        return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

def build_model(name: str, seq_len: int, pred_len: int, n_feat: int,
                use_residual: bool = True, **kwargs) -> nn.Module:
    """Instantiate a model by name. Pass use_residual=False for ablation."""
    name = name.lower()
    if name == "persistence":
        return PersistenceBaseline(pred_len)
    if name in ("seasonalnaive", "seasonal_naive", "snaive"):
        return SeasonalNaiveBaseline(pred_len, period=kwargs.get("period", 168))
    if name == "mlp":
        return MLPForecaster(seq_len, pred_len, n_feat,
                             use_residual=use_residual, **kwargs)
    if name == "lstm":
        return LSTMForecaster(pred_len, n_feat,
                              use_residual=use_residual, **kwargs)
    if name == "dlinear":
        return DLinear(seq_len, pred_len, n_feat,
                       use_residual=use_residual, **kwargs)
    if name in ("tcdlinear", "tc_dlinear", "gated_dlinear"):
        return TCGatedDLinear(seq_len, pred_len, n_feat,
                              use_residual=use_residual, **kwargs)
    if name == "patchtst":
        return PatchTST(seq_len, pred_len, n_feat,
                        use_residual=use_residual, **kwargs)
    if name == "itransformer":
        return ITransformer(seq_len, pred_len, n_feat,
                            use_residual=use_residual, **kwargs)
    if name == "dpmixer":
        return DPMixer(seq_len, pred_len, n_feat,
                       use_residual=use_residual, **kwargs)
    if name in ("tcdpmixer", "tc_dpmixer"):
        return TCDPMixer(seq_len, pred_len, n_feat,
                         use_residual=use_residual, **kwargs)
    if name in ("ssdpmixer", "ss_dpmixer"):
        return SSDPMixer(seq_len, pred_len, n_feat, **kwargs)
    if name in ("tcssdpmixer", "tcss_dpmixer", "tc_ss_dpmixer"):
        return TCSSDPMixer(seq_len, pred_len, n_feat, **kwargs)
    if name in ("tcspike", "tcspikeclassifier", "tc_spike"):
        # use_residual is intentionally ignored — the classifier forces it off.
        return TCSpikeClassifier(seq_len, pred_len, n_feat, **kwargs)
    raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    B, L, C, H = 8, 96, 26, 24
    x = torch.randn(B, L, C)

    print("With residual:")
    for name in ["persistence", "mlp", "lstm", "dlinear", "patchtst",
                 "itransformer", "dpmixer"]:
        m   = build_model(name, L, H, C, use_residual=True)
        out = m(x)
        assert out.shape == (B, H), f"{name}: {out.shape}"
        npar = sum(p.numel() for p in m.parameters())
        print(f"  {name:<14} out={tuple(out.shape)}  params={npar:,}")

    print("\nWithout residual (ablation):")
    for name in ["mlp", "lstm", "dlinear", "patchtst", "itransformer", "dpmixer"]:
        m   = build_model(name, L, H, C, use_residual=False)
        assert m(x).shape == (B, H)

    print("\n✓ models OK")
