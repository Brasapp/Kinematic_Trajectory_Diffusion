"""
diffusion_math.py
=================
Core mathematical engine for the Kinematic-Guided Trajectory Diffusion Model.

This module implements two primary components:

1.  CosineNoiseScheduler
    ---------------------
    Implements the cosine variance schedule from Nichol & Dhariwal (2021),
    "Improved Denoising Diffusion Probabilistic Models", and provides
    closed-form methods for the forward process (q-sampling) and the
    posterior distribution used in the reverse process.

2.  KinematicGuidance
    ------------------
    Implements a differentiable kinematic cost function J(s) that penalises
    physically implausible trajectories.  Its gradient is injected into the
    DDIM reverse step to steer generation towards smooth, mission-plausible
    UAV paths.

References
----------
- Ho et al. (2020)  "Denoising Diffusion Probabilistic Models"
  https://arxiv.org/abs/2006.11239
- Nichol & Dhariwal (2021)  "Improved Denoising Diffusion Probabilistic Models"
  https://arxiv.org/abs/2102.09672
- Song et al. (2020)  "Denoising Diffusion Implicit Models"
  https://arxiv.org/abs/2010.02502
- Dhariwal & Nichol (2021)  "Diffusion Models Beat GANs on Image Synthesis"
  https://arxiv.org/abs/2105.05233
"""

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# 1.  COSINE NOISE SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

class CosineNoiseScheduler:
    r"""Cosine variance schedule for DDPM / DDIM.

    The cosine schedule defines :math:`\bar{\alpha}_t` (cumulative signal
    power) as a smooth function of the timestep :math:`t`:

    .. math::

        \bar{\alpha}_t = \frac{f(t)}{f(0)}, \qquad
        f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)

    where :math:`s = 0.008` is a small offset that prevents
    :math:`\beta_t` from being too small near :math:`t = 0`.

    The individual noise variance at step :math:`t` is then:

    .. math::

        \beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}},
        \qquad \beta_t \in [\epsilon, 0.9999]

    All schedule tensors are pre-computed and cached on the given device for
    efficient look-up during training and sampling.

    Parameters
    ----------
    T : int
        Total number of diffusion timesteps (e.g. 1000).
    s : float
        Cosine schedule offset (default ``0.008``, per Nichol & Dhariwal 2021).
    device : str or torch.device
        Device on which to store the schedule tensors.

    Attributes
    ----------
    betas              : Tensor[T]   — :math:`\beta_t`
    alphas             : Tensor[T]   — :math:`\alpha_t = 1 - \beta_t`
    alpha_bars         : Tensor[T]   — :math:`\bar{\alpha}_t`
    alpha_bars_prev    : Tensor[T]   — :math:`\bar{\alpha}_{t-1}` (padded with 1 at index 0)
    posterior_variance : Tensor[T]   — :math:`\tilde{\beta}_t`
    posterior_log_var  : Tensor[T]   — :math:`\log \tilde{\beta}_t` (clipped)
    """

    def __init__(self, T: int = 1000, s: float = 0.008, device: str = "cpu"):
        self.T = T
        self.device = device

        # ── Compute f(t) on a fine integer grid [0, T] ──────────────────────
        # f(t) = cos^2( (t/T + s) / (1 + s) * pi/2 )
        t = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos(((t / T + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2

        # ── alpha_bar_t = f(t) / f(0) ────────────────────────────────────────
        alpha_bars_full = (f / f[0]).clamp(min=1e-8)  # shape [T+1]

        # ── beta_t = 1 - alpha_bar_t / alpha_bar_{t-1} ──────────────────────
        # Index 0 of alpha_bars_full = f(0)/f(0) = 1 (no noise yet)
        betas = 1.0 - alpha_bars_full[1:] / alpha_bars_full[:-1]
        betas = betas.clamp(1e-5, 0.9999).float()

        alphas = 1.0 - betas                       # shape [T]
        alpha_bars = alpha_bars_full[1:].float()   # shape [T]

        # alpha_bar_{t-1}: prepend 1.0 (since alpha_bar_0 = 1 by convention)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32),
                                     alpha_bars[:-1]])  # shape [T]

        # ── Posterior variance: ~beta_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # Used in the reverse DDPM step:
        #   q(x_{t-1} | x_t, x_0) ~ N(mu_tilde, tilde_beta_t * I)
        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        ).clamp(min=1e-20)

        # Store all tensors on the chosen device
        self.betas               = betas.to(device)
        self.alphas              = alphas.to(device)
        self.alpha_bars          = alpha_bars.to(device)
        self.alpha_bars_prev     = alpha_bars_prev.to(device)
        self.posterior_variance  = posterior_variance.to(device)
        self.posterior_log_var   = torch.log(posterior_variance).to(device)

    # ── Helper: extract values at batch timesteps ────────────────────────────
    def _extract(self, tensor: torch.Tensor, t: torch.Tensor,
                 x_shape: tuple) -> torch.Tensor:
        r"""Index ``tensor`` at positions ``t`` and reshape for broadcasting.

        Parameters
        ----------
        tensor : Tensor[T]
        t      : LongTensor[B]   — batch of timestep indices
        x_shape: tuple           — shape of the trajectory tensor, e.g. (B, F, L)

        Returns
        -------
        Tensor of shape ``(B, 1, 1)`` ready for element-wise multiplication.
        """
        out = tensor.gather(0, t)  # [B]
        # Reshape to [B, 1, 1, ...] with len(x_shape)-1 trailing singleton dims
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    # ── Forward process: sample x_t given x_0 ────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None):
        r"""Sample a noisy trajectory :math:`\mathbf{x}_t` from :math:`\mathbf{x}_0`.

        Implements the closed-form forward marginal:

        .. math::

            \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
                          + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon},
            \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})

        Parameters
        ----------
        x0    : Tensor[B, F, L]  — clean trajectory (batch, features, length)
        t     : LongTensor[B]    — integer timestep indices
        noise : Tensor[B, F, L]  — optional pre-sampled Gaussian noise

        Returns
        -------
        x_t   : Tensor[B, F, L] — noisy trajectory at timestep t
        noise : Tensor[B, F, L] — the actual noise added (needed for training loss)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar       = self._extract(self.alpha_bars.sqrt(),       t, x0.shape)
        sqrt_one_minus_alpha = self._extract((1 - self.alpha_bars).sqrt(), t, x0.shape)

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    # ── Predict x_0 from (x_t, t, eps_pred) ─────────────────────────────────
    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                            eps: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct :math:`\hat{\mathbf{x}}_0` from a predicted noise :math:`\hat{\boldsymbol{\epsilon}}`.

        Inverting the forward marginal:

        .. math::

            \hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t
                - \sqrt{1 - \bar{\alpha}_t}\,\hat{\boldsymbol{\epsilon}}}
                {\sqrt{\bar{\alpha}_t}}

        Parameters
        ----------
        x_t : Tensor[B, F, L] — noisy trajectory at timestep t
        t   : LongTensor[B]
        eps : Tensor[B, F, L] — predicted noise from the score network

        Returns
        -------
        x0_pred : Tensor[B, F, L]
        """
        sqrt_alpha_bar       = self._extract(self.alpha_bars.sqrt(),       t, x_t.shape)
        sqrt_one_minus_alpha = self._extract((1 - self.alpha_bars).sqrt(), t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha_bar

    # ── Posterior mean (used in DDPM ancestral sampling) ─────────────────────
    def q_posterior_mean(self, x0: torch.Tensor, x_t: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        r"""Compute the posterior mean :math:`\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)`.

        .. math::

            \tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}
                                 {1 - \bar{\alpha}_t}\,\mathbf{x}_0
                          + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}
                                 {1 - \bar{\alpha}_t}\,\mathbf{x}_t

        Parameters
        ----------
        x0  : Tensor[B, F, L] — reconstructed clean trajectory
        x_t : Tensor[B, F, L] — noisy trajectory at timestep t
        t   : LongTensor[B]

        Returns
        -------
        mu_tilde : Tensor[B, F, L]
        """
        ab_t    = self._extract(self.alpha_bars,      t, x0.shape)
        ab_prev = self._extract(self.alpha_bars_prev, t, x0.shape)
        beta_t  = self._extract(self.betas,           t, x0.shape)
        alpha_t = self._extract(self.alphas,          t, x0.shape)

        coef_x0 = ab_prev.sqrt() * beta_t / (1.0 - ab_t)
        coef_xt = alpha_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t)

        return coef_x0 * x0 + coef_xt * x_t


# ──────────────────────────────────────────────────────────────────────────────
# 2.  KINEMATIC GUIDANCE
# ──────────────────────────────────────────────────────────────────────────────

class KinematicGuidance:
    r"""Kinematic guidance applied to the predicted clean trajectory.

    All terms are evaluated in physical units (radians, metres, seconds) and
    calibrated on the real AUTO data (see ``calibrate_guidance`` in sample.py).

    **Pitch-climb coupling** — climb rate should follow the attitude:

    .. math::

        r_i = \frac{1}{\sigma_c}\left(\frac{z_{i+1}-z_i}{\Delta t} - g\sin\theta_i - b\right),
        \qquad J_{pitch} = \sum_i r_i^2

    :math:`g` (effective speed) and :math:`b` (climb bias, i.e. trim / angle of
    attack) come from a least-squares fit on the real data; :math:`\sigma_c` is
    the residual RMS of that fit.

    **Roll chatter** — penalises roll *acceleration*, not roll rate, so real
    turn entries are kept but step-to-step jitter is removed:

    .. math::

        J_{roll} = \sum_i \left(\frac{\phi_{i+1}-2\phi_i+\phi_{i-1}}{\sigma_{\ddot\phi}}\right)^2

    **Attitude limits** — quadratic penalty outside the real envelope
    (in units of the channel std).

    **Update rule** — one preconditioned gradient step per term and channel:

    .. math::

        x_c \leftarrow x_c - w\,\bar\alpha_t \sum_k \lambda_k
                \frac{\partial J_k / \partial x_c}{L_{k,c}}

    where :math:`L_{k,c}` is the Lipschitz constant of :math:`\nabla J_k` in
    channel :math:`c`.  With :math:`w\sum_k\lambda_k \le 1` a step can never
    overshoot, so the guidance cannot make the trajectory oscillate.  The
    :math:`\bar\alpha_t` factor keeps guidance off while :math:`\hat x_0` is
    still mostly noise.  Optionally altitude is re-anchored so the window
    starts at 0 m (the training data is altitude relative to window start).
    """

    def __init__(
        self,
        guidance_scale: float = 0.0,
        lambda_roll: float = 1.0,
        lambda_pitch: float = 1.0,
        lambda_bound: float = 1.0,
        roll_channel: int = 0,
        pitch_channel: int = 1,
        alt_channel: int = 2,
        dt: float = 0.1,
        climb_gain: float = 30.0,
        climb_bias: float = 0.0,
        climb_resid_std: float = 3.0,
        roll_d2_std: float = 0.06,
        max_roll_deg: float = 90.0,
        min_pitch_deg: float = -20.0,
        max_pitch_deg: float = 40.0,
        anchor_alt: bool = True,
        pitch_drives_alt: bool = True,
        iters: int = 1,
        means: torch.Tensor = None,
        stds: torch.Tensor = None,
    ):
        self.w = guidance_scale
        self.lam = {"roll": lambda_roll, "pitch": lambda_pitch, "bound": lambda_bound}
        self.r_ch, self.p_ch, self.a_ch = roll_channel, pitch_channel, alt_channel
        self.dt = dt
        self.g, self.b, self.sig_c = climb_gain, climb_bias, climb_resid_std
        self.sig_r2 = roll_d2_std
        self.max_roll = math.radians(max_roll_deg)
        self.min_pitch = math.radians(min_pitch_deg)
        self.max_pitch = math.radians(max_pitch_deg)
        self.anchor_alt = anchor_alt
        self.iters = iters
        self.means = means if means is not None else torch.zeros(3)
        self.stds = stds if stds is not None else torch.ones(3)

        s = self.stds
        sr, sp, sa = float(s[self.r_ch]), float(s[self.p_ch]), float(s[self.a_ch])
        # Lipschitz constants of each term's gradient, per normalised channel.
        # Second difference: ||D2^T D2|| <= 16; first difference: ||D1^T D1|| <= 4.
        self.L = {
            "roll":  {self.r_ch: 2 * 16 * (sr / self.sig_r2) ** 2},
            "pitch": {self.a_ch: 2 * 4 * (sa / (self.dt * self.sig_c)) ** 2,
                      self.p_ch: 2 * (self.g * sp / self.sig_c) ** 2},
            "bound": {self.r_ch: 2.0, self.p_ch: 2.0},
        }
        if pitch_drives_alt:
            # Pitch is the cleaner channel: let the coupling term correct only
            # altitude, otherwise altitude noise is copied into pitch.
            del self.L["pitch"][self.p_ch]

    # ── Physical-unit view of a normalised tensor ────────────────────────────
    def _phys(self, x):
        mean = self.means.to(x.device, x.dtype).view(1, -1, 1)
        std = self.stds.to(x.device, x.dtype).view(1, -1, 1)
        return x * std + mean

    # ── Individual cost terms (summed over time, per sample) ────────────────
    def cost_terms(self, x: torch.Tensor) -> dict:
        """Return {name: Tensor[B]} cost terms for a normalised batch [B, F, L]."""
        xp = self._phys(x)
        roll, pitch, alt = xp[:, self.r_ch], xp[:, self.p_ch], xp[:, self.a_ch]

        d2 = roll[:, 2:] - 2 * roll[:, 1:-1] + roll[:, :-2]
        J_roll = ((d2 / self.sig_r2) ** 2).sum(-1)

        climb = (alt[:, 1:] - alt[:, :-1]) / self.dt
        r = (climb - self.g * torch.sin(pitch[:, :-1]) - self.b) / self.sig_c
        J_pitch = (r ** 2).sum(-1)

        sr = float(self.stds[self.r_ch]); sp = float(self.stds[self.p_ch])
        relu = torch.nn.functional.relu
        J_bound = ((relu(roll.abs() - self.max_roll) / sr) ** 2).sum(-1) \
            + ((relu(self.min_pitch - pitch) / sp) ** 2).sum(-1) \
            + ((relu(pitch - self.max_pitch) / sp) ** 2).sum(-1)
        return {"roll": J_roll, "pitch": J_pitch, "bound": J_bound}

    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted total cost per sample, Tensor[B]."""
        terms = self.cost_terms(x)
        return sum(self.lam[k] * v for k, v in terms.items())

    # ── Apply guidance to a predicted x_0 ────────────────────────────────────
    def apply(self, x0_pred: torch.Tensor, alpha_bar: float = 1.0) -> torch.Tensor:
        """Return the guided x̂_0 (normalised, [B, F, L])."""
        x = x0_pred
        w_t = self.w * float(alpha_bar)
        if w_t > 0:
            for _ in range(self.iters):
                step = torch.zeros_like(x)
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    terms = self.cost_terms(xg)
                    for name, J in terms.items():
                        if self.lam[name] == 0:
                            continue
                        (g,) = torch.autograd.grad(J.sum(), xg, retain_graph=True)
                        for c, Lc in self.L[name].items():
                            step[:, c] += self.lam[name] * g[:, c] / Lc
                x = (x - w_t * step).detach()
        if self.anchor_alt:
            # physical alt[0] -> 0  <=>  normalised alt[0] -> -mean/std
            a0 = -float(self.means[self.a_ch]) / float(self.stds[self.a_ch])
            x = x.clone()
            x[:, self.a_ch] -= x[:, self.a_ch, :1] - a0
        return x
