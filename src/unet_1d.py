"""
unet_1d.py
==========
Lightweight 1D U-Net score network for trajectory diffusion.

Architecture overview
---------------------
The network maps a noisy trajectory ``x_t`` and a scalar diffusion timestep
``t`` to the predicted noise ``ε_θ(x_t, t)``.

Temporal context is injected via **sinusoidal time embeddings** (Vaswani et al.
2017) followed by a two-layer MLP.  Each residual convolutional block uses
**FiLM** (Feature-wise Linear Modulation) conditioning: the time embedding is
projected to per-channel ``(scale, shift)`` pairs that modulate the feature
maps after GroupNorm.

Input / output shapes::

    x_t  : [B, 3, L]  — noisy trajectory [roll, pitch, alt]
    t    : [B]         — integer diffusion timestep (0 … T-1)
    ↓
    ε_θ  : [B, 3, L]  — predicted noise

The architecture is intentionally compact (no attention) since UAV trajectories
are short 1D sequences — not high-resolution images.

References
----------
- Ho et al. (2020)  "Denoising Diffusion Probabilistic Models"
- Ronneberger et al. (2015)  "U-Net: Convolutional Networks for Biomedical
  Image Segmentation"
- Perez et al. (2018)  "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SINUSOIDAL TIME EMBEDDING
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    r"""Sinusoidal positional encoding adapted for scalar diffusion timesteps.

    For a timestep :math:`t` and embedding dimension :math:`d`, the
    :math:`i`-th component is:

    .. math::

        \text{emb}(t)_{2i}   = \sin\!\left(\frac{t}{10000^{2i/d}}\right),
        \qquad
        \text{emb}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)

    The raw sinusoidal vector is then passed through a two-layer MLP with SiLU
    activations to produce the final time-conditioning vector:

    .. math::

        \mathbf{e}_t = W_2 \, \sigma(W_1 \, \text{emb}(t) + b_1) + b_2

    Parameters
    ----------
    dim : int
        Output dimension of the time embedding MLP.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # MLP: d → 4d → d  (as in Ho et al. 2020)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : LongTensor[B]  — integer diffusion timesteps

        Returns
        -------
        emb : Tensor[B, dim]
        """
        half = self.dim // 2
        # Frequency bands: 10000^{2i/d} for i in [0, half)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )  # [half]

        # Outer product: t[:, None] * freqs[None, :]  →  [B, half]
        args = t[:, None].float() * freqs[None, :]

        # Concatenate sin and cos: [B, dim]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        return self.mlp(emb)  # [B, dim]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    r"""1D Residual Convolutional Block with FiLM time conditioning.

    Structure::

        x ──┬── Conv1d ── GroupNorm ── FiLM(t) ── SiLU ── Conv1d ── GroupNorm ──┬── out
            └──────────────── proj (1×1 if in≠out) ─────────────────────────────┘

    FiLM modulation applies per-channel ``scale`` and ``shift`` derived from
    the time embedding:

    .. math::

        \mathbf{h}' = (1 + \gamma_t) \odot \text{GroupNorm}(\mathbf{h})
                      + \delta_t

    where :math:`(\gamma_t, \delta_t)` are learned linear projections of
    :math:`\mathbf{e}_t`.

    Parameters
    ----------
    in_ch    : int   — input channels
    out_ch   : int   — output channels
    t_emb_dim: int   — dimension of the time embedding
    groups   : int   — number of groups for GroupNorm (default 8)
    """

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, groups: int = 8):
        super().__init__()

        # Clamp groups so it never exceeds channel count
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)

        self.norm1  = nn.GroupNorm(g1, in_ch)
        self.conv1  = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2  = nn.GroupNorm(g2, out_ch)
        self.conv2  = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act    = nn.SiLU()

        # FiLM projection: t_emb → (gamma, delta) for out_ch channels each
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch * 2),  # split → scale + shift
        )

        # Residual projection (1×1 conv) if channel sizes differ
        self.residual_proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x     : Tensor[B, in_ch, L]
        t_emb : Tensor[B, t_emb_dim]

        Returns
        -------
        out : Tensor[B, out_ch, L]
        """
        residual = self.residual_proj(x)

        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # FiLM: compute scale and shift from time embedding
        # film_params: [B, out_ch*2] → split to [B, out_ch] each
        film_params = self.time_proj(t_emb)
        scale, shift = film_params.chunk(2, dim=-1)
        # Reshape for broadcasting: [B, out_ch, 1]
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        h = self.act(self.norm2(h) * (1.0 + scale) + shift)
        h = self.conv2(h)

        return h + residual


class DownBlock(nn.Module):
    """Encoder block: ResBlock1D → save skip → MaxPool1d(2).

    Parameters
    ----------
    in_ch     : int
    out_ch    : int
    t_emb_dim : int
    """

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int):
        super().__init__()
        self.res  = ResBlock1D(in_ch, out_ch, t_emb_dim)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        Returns
        -------
        pooled : Tensor[B, out_ch, L//2]  — passed to next encoder stage
        skip   : Tensor[B, out_ch, L]     — saved for skip connection
        """
        skip   = self.res(x, t_emb)   # [B, out_ch, L]
        pooled = self.pool(skip)       # [B, out_ch, L//2]
        return pooled, skip


class UpBlock(nn.Module):
    """Decoder block: ConvTranspose1d (×2) → concat skip → ResBlock1D.

    Parameters
    ----------
    in_ch     : int   — channels from the deeper stage (before concat)
    skip_ch   : int   — channels of the skip tensor
    out_ch    : int   — output channels
    t_emb_dim : int
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_emb_dim: int):
        super().__init__()
        self.up  = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.res = ResBlock1D(in_ch + skip_ch, out_ch, t_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor[B, in_ch, L//2]
        skip : Tensor[B, skip_ch, L]
        t_emb: Tensor[B, t_emb_dim]

        Returns
        -------
        out  : Tensor[B, out_ch, L]
        """
        x = self.up(x)                            # [B, in_ch, L]

        # Trim or pad spatial dimension if needed (avoids off-by-one errors)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear",
                              align_corners=False)

        x = torch.cat([x, skip], dim=1)           # [B, in_ch + skip_ch, L]
        return self.res(x, t_emb)                 # [B, out_ch, L]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  FULL 1D U-NET
# ──────────────────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    r"""1D U-Net for predicting noise :math:`\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)`.

    The network takes a noisy trajectory sequence and the diffusion timestep,
    and returns the predicted noise at every position of the sequence.

    Architecture (default: depth=3, base_channels=64)::

        Input [B, 7, L]
          ↓ stem Conv1d(7, 64)
          ↓ DownBlock(64 → 64)   skip₁ [B, 64,  L]
          ↓ DownBlock(64 → 128)  skip₂ [B, 128, L/2]
          ↓ DownBlock(128 → 256) skip₃ [B, 256, L/4]
          ↓ Bottleneck ResBlock(256 → 256)
          ↓ UpBlock(256, skip₃=256, → 128)
          ↓ UpBlock(128, skip₂=128, → 64)
          ↓ UpBlock(64,  skip₁=64,  → 64)
          ↓ head Conv1d(64, 7)
        Output [B, 7, L]  — predicted noise ε_θ

    Parameters
    ----------
    in_channels   : int   — number of trajectory features (default 7)
    base_channels : int   — width of the first U-Net level (default 64)
    depth         : int   — number of encoder/decoder stages (default 3)
    time_emb_dim  : int   — dimension of the time embedding (default 128)
    """

    def __init__(
        self,
        in_channels: int = 7,
        base_channels: int = 64,
        depth: int = 3,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.base_channels = base_channels
        self.depth        = depth

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # ── Channel widths at each U-Net level ────────────────────────────────
        # e.g. depth=3, base=64 → [64, 128, 256]
        ch = [base_channels * (2 ** i) for i in range(depth)]

        # ── Stem: project input features into channel space ───────────────────
        self.stem = nn.Conv1d(in_channels, ch[0], kernel_size=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        in_ch = ch[0]
        for out_ch in ch:
            self.down_blocks.append(DownBlock(in_ch, out_ch, time_emb_dim))
            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ResBlock1D(ch[-1], ch[-1], time_emb_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        # ch (reversed): [256, 128, 64]  for depth=3
        self.up_blocks = nn.ModuleList()
        rev_ch = list(reversed(ch))          # deepest → shallowest
        decoder_in = ch[-1]
        for i, skip_ch in enumerate(rev_ch):
            out_ch = rev_ch[i + 1] if i + 1 < len(rev_ch) else rev_ch[-1]
            self.up_blocks.append(
                UpBlock(decoder_in, skip_ch, out_ch, time_emb_dim)
            )
            decoder_in = out_ch

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.GroupNorm(min(8, decoder_in), decoder_in),
            nn.SiLU(),
            nn.Conv1d(decoder_in, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Forward pass: predict noise :math:`\hat{\boldsymbol{\epsilon}}_\theta`.

        Parameters
        ----------
        x : Tensor[B, F, L]  — noisy trajectory ``x_t``
        t : LongTensor[B]    — integer diffusion timesteps

        Returns
        -------
        eps_pred : Tensor[B, F, L]  — predicted noise
        """
        # Time embedding: [B] → [B, time_emb_dim]
        t_emb = self.time_emb(t)

        # Stem projection
        h = self.stem(x)  # [B, ch[0], L]

        # Encoder: collect skip connections
        skips = []
        for block in self.down_blocks:
            h, skip = block(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder: pop skip connections in reverse order (deepest first)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = block(h, skip, t_emb)

        return self.head(h)  # [B, F, L]


class MirrorSymmetric(nn.Module):
    r"""Make a noise predictor exactly left/right symmetric.

    With :math:`M` the mirror (sign flip of the roll channel),

    .. math::

        \hat\epsilon(x, t) = \tfrac12\left[\epsilon_\theta(x, t)
                                + M\,\epsilon_\theta(Mx, t)\right]

    satisfies :math:`\hat\epsilon(Mx, t) = M\hat\epsilon(x, t)`.  Because the
    starting noise is symmetric too, the sampled distribution is then exactly
    mirror-symmetric: left and right turns are equally likely, whatever the
    network learned.  Both halves run as one batch of size 2B.
    """

    def __init__(self, net: nn.Module, mirror_channels, in_channels: int):
        super().__init__()
        self.net = net
        sign = torch.ones(1, in_channels, 1)
        for c in mirror_channels:
            sign[0, c] = -1.0
        self.register_buffer("sign", sign, persistent=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = self.net(torch.cat([x, self.sign * x]), torch.cat([t, t]))
        return 0.5 * (out[:B] + self.sign * out[B:])


def build_model(config: dict) -> nn.Module:
    """UNet1D from the ``model:`` config, wrapped in MirrorSymmetric when
    ``model.mirror_symmetric`` and ``data.mirror_roll`` are both on."""
    m = config["model"]
    net = UNet1D(
        in_channels=m["in_channels"],
        base_channels=m["base_channels"],
        depth=m["depth"],
        time_emb_dim=m["time_emb_dim"],
    )
    names = list(config["data"].get("feature_names", ["roll", "pitch", "alt"]))
    if m.get("mirror_symmetric", False) and config["data"].get("mirror_roll", False) and "roll" in names:
        return MirrorSymmetric(net, [names.index("roll")], m["in_channels"])
    return net


def core_module(model: nn.Module) -> nn.Module:
    """The bare UNet1D (checkpoints always store its weights)."""
    return model.net if isinstance(model, MirrorSymmetric) else model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, F, L = 4, 3, 256
    model = UNet1D(in_channels=F, base_channels=64, depth=3, time_emb_dim=128)
    x = torch.randn(B, F, L)
    t = torch.randint(0, 1000, (B,))
    out = model(x, t)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}  (expected {x.shape})")
    print(f"Params: {n_params:,}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("UNet1D sanity check PASSED ✓")
