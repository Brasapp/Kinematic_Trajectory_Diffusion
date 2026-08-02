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
    r"""Differentiable kinematic cost function for guided trajectory sampling.

    The cost function :math:`J(\mathbf{s})` penalises two types of physically
    unnatural motion in a UAV trajectory:

    **Term 1 — Yaw-rate penalty** (sharp, unnatural turns)

    .. math::

        J_{\psi}(\mathbf{s}) = \sum_{i=1}^{T-1} (\psi_{i+1} - \psi_i)^2

    where :math:`\psi_i` is the yaw angle (radians) at step :math:`i`.

    **Term 2 — Velocity-change penalty** (abrupt accelerations)

    .. math::

        J_{v}(\mathbf{s}) = \sum_{i=1}^{T-1}
            \left\|\mathbf{v}_{i+1} - \mathbf{v}_i\right\|^2

    where :math:`\mathbf{v}_i = (v_N, v_E, v_D)_i`.

    **Combined cost**

    .. math::

        J(\mathbf{s}) = J_{\psi}(\mathbf{s}) + \lambda \, J_{v}(\mathbf{s})

    During sampling, the gradient :math:`\nabla_{\mathbf{x}_t} J` is computed
    via ``torch.autograd.grad`` and subtracted from the predicted clean
    trajectory :math:`\hat{\mathbf{x}}_0` with a guidance scale :math:`w`:

    .. math::

        \hat{\mathbf{x}}_0^{\text{guided}}
            = \hat{\mathbf{x}}_0
            - w \cdot \nabla_{\hat{\mathbf{x}}_0} J(\hat{\mathbf{x}}_0)

    Parameters
    ----------
    guidance_scale : float
        :math:`w` — strength of guidance.  0 = no guidance, larger = smoother.
    lambda_vel : float
        :math:`\lambda` — relative weight of the velocity-change penalty.
    yaw_channel : int
        Feature channel index of the yaw angle in the trajectory tensor.
        Default ``6`` for feature order ``[N, E, D, vN, vE, vD, yaw]``.
    vel_channels : slice
        Feature channel slice of velocity components ``[vN, vE, vD]``.
        Default ``slice(3, 6)``.
    """

    def __init__(
        self,
        guidance_scale: float = 2.0,
        lambda_vel: float = 0.1,
        lambda_roll: float = 1.0,
        lambda_pitch: float = 0.5,
        lambda_coord: float = 2.0,
        yaw_channel: int = 6,
        roll_channel: int = 7,
        pitch_channel: int = 8,
        vel_channels: slice = slice(3, 6),
        dt: float = 0.1,
    ):
        self.w           = guidance_scale
        self.lam_v       = lambda_vel
        self.lam_r       = lambda_roll
        self.lam_p       = lambda_pitch
        self.lam_c       = lambda_coord
        self.yaw_ch      = yaw_channel
        self.roll_ch     = roll_channel
        self.pitch_ch    = pitch_channel
        self.vel_sl      = vel_channels
        self.dt          = dt
        self.g           = 9.81

    # ── Compute J(s) ─────────────────────────────────────────────────────────
    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the kinematic cost :math:`J(\mathbf{s})`.

        Parameters
        ----------
        x : Tensor[B, F, L]
            Trajectory tensor.  ``x[:, yaw_channel, :]`` contains yaw;
            ``x[:, vel_channels, :]`` contains (vN, vE, vD).

        Returns
        -------
        cost : scalar Tensor
            Mean kinematic cost across the batch.
        """
        # ── Velocity-change cost ─────────────────────────────────────────────
        vel = x[:, self.vel_sl, :]
        delta_v = vel[:, :, 1:] - vel[:, :, :-1]
        J_vel = (delta_v ** 2).mean()

        # ── Roll-rate and Pitch-rate cost ────────────────────────────────────
        roll = x[:, self.roll_ch, :]
        delta_roll = roll[:, 1:] - roll[:, :-1]
        J_roll = (delta_roll ** 2).mean()

        pitch = x[:, self.pitch_ch, :]
        delta_pitch = pitch[:, 1:] - pitch[:, :-1]
        J_pitch = (delta_pitch ** 2).mean()

        # ── Coordinated-turn constraint ──────────────────────────────────────
        yaw = x[:, self.yaw_ch, :]
        delta_yaw = yaw[:, 1:] - yaw[:, :-1]

        vel_i = vel[:, :, :-1]
        speed_i = torch.norm(vel_i, dim=1) + 1e-4  # [B, L-1], guard against div/0
        roll_i = roll[:, :-1]

        # constraint: delta_psi = (g * dt / V) * tan(phi)
        coord_err = delta_yaw - (self.g * self.dt / speed_i) * torch.tan(roll_i)
        J_coord = (coord_err ** 2).mean()

        # ── Combined cost ────────────────────────────────────────────────────
        J = self.lam_r * J_roll + self.lam_p * J_pitch + self.lam_v * J_vel + self.lam_c * J_coord
        return J

    # ── Compute gradient ∇J via autograd ─────────────────────────────────────
    def compute_gradient(self, x_hat: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\nabla_{\hat{\mathbf{x}}_0} J(\hat{\mathbf{x}}_0)`.

        Uses ``torch.autograd.grad`` so that no computational graph is retained
        after the call.  The input tensor is cloned with ``requires_grad=True``
        to avoid mutating the diffusion graph.

        Parameters
        ----------
        x_hat : Tensor[B, F, L]
            Predicted clean trajectory (detached from the main graph).

        Returns
        -------
        grad : Tensor[B, F, L]  (detached)
        """
        with torch.enable_grad():
            x = x_hat.detach().requires_grad_(True)
            cost = self.compute_cost(x)
            (grad,) = torch.autograd.grad(cost, x)
        return grad.detach()

    # ── Apply guidance to a predicted x_0 ────────────────────────────────────
    def apply(self, x0_pred: torch.Tensor) -> torch.Tensor:
        r"""Return the kinematic-guided :math:`\hat{\mathbf{x}}_0`.

        .. math::

            \hat{\mathbf{x}}_0^{\text{guided}}
                = \hat{\mathbf{x}}_0 - w \cdot \nabla_{\hat{\mathbf{x}}_0} J

        Parameters
        ----------
        x0_pred : Tensor[B, F, L] — predicted clean trajectory

        Returns
        -------
        x0_guided : Tensor[B, F, L]
        """
        grad = self.compute_gradient(x0_pred)
        return x0_pred - self.w * grad
