from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


class SigmaScan:
    @staticmethod
    def per_pixel_loss(amps, mus, sigmas, height_axis, profile):
        safe_s2 = 2.0 * jnp.maximum(sigmas, 1e-6) ** 2
        diff    = height_axis[None, :] - mus[:, None]
        expon   = jnp.clip(-(diff ** 2) / safe_s2[:, None], -100.0, 0.0)
        pred    = (amps[:, None] * jnp.exp(expon)).sum(axis=0)
        mse     = jnp.mean((pred - profile) ** 2)

        return mse

    @staticmethod
    def adam_scan(
        batched_vg  ,
        batched_loss,
        amps_init    : jnp.ndarray,
        mus_init     : jnp.ndarray,
        sigmas_init  : jnp.ndarray,
        height_axis  : jnp.ndarray,
        profiles     : jnp.ndarray,
        amp_mask     : jnp.ndarray,
        mu_mask      : jnp.ndarray,
        sigma_mask   : jnp.ndarray,
        mu_lower     : jnp.ndarray,
        mu_upper     : jnp.ndarray,
        sigma_lower  : jnp.ndarray,
        sigma_upper  : jnp.ndarray,
        n_steps      : int,
        lr           : float,
        b1           : float,
        b2           : float,
    ) -> tuple:
        b1_         = jnp.float32(b1)
        b2_         = jnp.float32(b2)
        eps         = jnp.float32(1e-8)
        lr_         = jnp.float32(lr)
        amp_mask_   = jnp.float32(amp_mask)
        mu_mask_    = jnp.float32(mu_mask)
        sigma_mask_ = jnp.float32(sigma_mask)

        a = jnp.maximum(amps_init.astype(jnp.float32),   0.0)
        u = jnp.clip(   mus_init.astype(jnp.float32),    mu_lower,    mu_upper)
        s = jnp.clip(   sigmas_init.astype(jnp.float32), sigma_lower, sigma_upper)

        m_a = jnp.zeros_like(a)
        v_a = jnp.zeros_like(a)
        m_u = jnp.zeros_like(u)
        v_u = jnp.zeros_like(u)
        m_s = jnp.zeros_like(s)
        v_s = jnp.zeros_like(s)

        def _adam(p, m_, v_, g, tf):
            m_ = b1_ * m_ + (1.0 - b1_) * g
            v_ = b2_ * v_ + (1.0 - b2_) * g * g
            p  = p - lr_ * (m_ / (1.0 - b1_ ** tf)) / (jnp.sqrt(v_ / (1.0 - b2_ ** tf)) + eps)
            return p, m_, v_

        def _step(carry, t):
            a_, u_, s_, m_a_, v_a_, m_u_, v_u_, m_s_, v_s_ = carry
            _, (g_a, g_u, g_s) = batched_vg(a_, u_, s_, height_axis, profiles)

            g_a = g_a * amp_mask_
            g_u = g_u * mu_mask_
            g_s = g_s * sigma_mask_
            tf  = t.astype(jnp.float32) + 1.0

            a_, m_a_, v_a_ = _adam(a_, m_a_, v_a_, g_a, tf)
            u_, m_u_, v_u_ = _adam(u_, m_u_, v_u_, g_u, tf)
            s_, m_s_, v_s_ = _adam(s_, m_s_, v_s_, g_s, tf)

            a_ = jnp.maximum(a_, 0.0)
            u_ = jnp.clip(u_, mu_lower,    mu_upper)
            s_ = jnp.clip(s_, sigma_lower, sigma_upper)

            return (a_, u_, s_, m_a_, v_a_, m_u_, v_u_, m_s_, v_s_), None

        carry0                 = (a, u, s, m_a, v_a, m_u, v_u, m_s, v_s)
        (a_f, u_f, s_f, *_), _ = jax.lax.scan(_step, carry0, jnp.arange(n_steps))
        mse_f                  = batched_loss(a_f, u_f, s_f, height_axis, profiles)

        return a_f, u_f, s_f, mse_f


class SigmaAdamKernel:
    def __init__(self) -> None:
        batched_vg   = jax.vmap(jax.value_and_grad(SigmaScan.per_pixel_loss, argnums=(0, 1, 2)), in_axes=(0, 0, 0, None, 0))
        batched_loss = jax.vmap(SigmaScan.per_pixel_loss, in_axes=(0, 0, 0, None, 0))
        self._run    = self._build(batched_vg, batched_loss)

    @staticmethod
    def _build(batched_vg, batched_loss):
        @partial(jax.jit, static_argnames=("n_steps", "lr", "b1", "b2"))
        def _run(
            amps_init   : jnp.ndarray,
            mus_init    : jnp.ndarray,
            sigmas_init : jnp.ndarray,
            height_axis : jnp.ndarray,
            profiles    : jnp.ndarray,
            amp_mask    : jnp.ndarray,
            mu_mask     : jnp.ndarray,
            sigma_mask  : jnp.ndarray,
            mu_lower    : jnp.ndarray,
            mu_upper    : jnp.ndarray,
            sigma_lower : jnp.ndarray,
            sigma_upper : jnp.ndarray,
            n_steps     : int   = 2000,
            lr          : float = 1e-2,
            b1          : float = 0.9,
            b2          : float = 0.999,
        ) -> tuple:
            return SigmaScan.adam_scan(batched_vg, batched_loss, amps_init, mus_init, sigmas_init, height_axis, profiles, amp_mask, mu_mask, sigma_mask, mu_lower, mu_upper, sigma_lower, sigma_upper, n_steps, lr, b1, b2)
        return _run

    def __call__(
        self,
        amps_init, mus_init, sigmas_init, height_axis, profiles,
        amp_mask, mu_mask, sigma_mask, mu_lower, mu_upper, sigma_lower, sigma_upper,
        n_steps=2000, lr=1e-2, b1=0.9, b2=0.999,
    ):
        return self._run(
            amps_init, mus_init, sigmas_init, height_axis, profiles,
            amp_mask, mu_mask, sigma_mask, mu_lower, mu_upper, sigma_lower, sigma_upper,
            n_steps, lr, b1, b2,
        )


class PmapSigmaAdamKernel:
    def __init__(self, devices: list) -> None:
        self._n_devices = len(devices)
        batched_vg      = jax.vmap(jax.value_and_grad(SigmaScan.per_pixel_loss, argnums=(0, 1, 2)), in_axes=(0, 0, 0, None, 0))
        batched_loss    = jax.vmap(SigmaScan.per_pixel_loss, in_axes=(0, 0, 0, None, 0))
        self._run       = self._build(batched_vg, batched_loss, devices)

    @staticmethod
    def _build(batched_vg, batched_loss, devices):
        def _run_on_device(
            amps_init   : jnp.ndarray,
            mus_init    : jnp.ndarray,
            sigmas_init : jnp.ndarray,
            height_axis : jnp.ndarray,
            profiles    : jnp.ndarray,
            amp_mask    : jnp.ndarray,
            mu_mask     : jnp.ndarray,
            sigma_mask  : jnp.ndarray,
            mu_lower    : jnp.ndarray,
            mu_upper    : jnp.ndarray,
            sigma_lower : jnp.ndarray,
            sigma_upper : jnp.ndarray,
            n_steps     : int   = 2000,
            lr          : float = 1e-2,
            b1          : float = 0.9,
            b2          : float = 0.999,
        ) -> tuple:
            return SigmaScan.adam_scan(batched_vg, batched_loss, amps_init, mus_init, sigmas_init, height_axis, profiles, amp_mask, mu_mask, sigma_mask, mu_lower, mu_upper, sigma_lower, sigma_upper, n_steps, lr, b1, b2)

        return jax.pmap(
            _run_on_device,
            in_axes                    = (0, 0, 0, None, 0, None, None, None, None, None, None, None),
            static_broadcasted_argnums = (12, 13, 14, 15),
            devices                    = devices,
        )

    def __call__(
        self,
        amps_init, mus_init, sigmas_init, height_axis, profiles,
        amp_mask, mu_mask, sigma_mask, mu_lower, mu_upper, sigma_lower, sigma_upper,
        n_steps=2000, lr=1e-2, b1=0.9, b2=0.999,
    ):
        n, K = sigmas_init.shape
        H    = profiles.shape[1]
        D    = self._n_devices
        pad  = (-n) % D

        if pad > 0:
            z_K         = jnp.zeros((pad, K), dtype=jnp.float32)
            z_H         = jnp.zeros((pad, H), dtype=jnp.float32)
            amps_init   = jnp.concatenate([amps_init,   z_K], axis=0)
            mus_init    = jnp.concatenate([mus_init,    z_K], axis=0)
            sigmas_init = jnp.concatenate([sigmas_init, z_K], axis=0)
            profiles    = jnp.concatenate([profiles,    z_H], axis=0)

        n_pad = n + pad
        shard = n_pad // D

        out_a, out_u, out_s, out_e = self._run(
            amps_init  .reshape(D, shard, K),
            mus_init   .reshape(D, shard, K),
            sigmas_init.reshape(D, shard, K),
            height_axis,
            profiles   .reshape(D, shard, H),
            amp_mask,
            mu_mask,
            sigma_mask,
            mu_lower,
            mu_upper,
            sigma_lower,
            sigma_upper,
            n_steps, lr, b1, b2,
        )

        out_a = out_a.reshape(n_pad, K)[:n]
        out_u = out_u.reshape(n_pad, K)[:n]
        out_s = out_s.reshape(n_pad, K)[:n]
        out_e = out_e.reshape(n_pad)[:n]

        return out_a, out_u, out_s, out_e
