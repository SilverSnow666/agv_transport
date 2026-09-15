"""Inference-only zero-mean, bounded residual action mapping (no environment edits)."""

import torch


def project_zero_mean(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center bounded policy actions, then radially scale only if necessary.

    Independent clipping would reintroduce a common component. Uniform scaling
    preserves differential direction but attenuates its magnitude when active;
    callers must report that confound. The input is not modified.
    """
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError("Expected [num_envs, 3] actions")
    if not torch.isfinite(actions).all() or torch.any(torch.abs(actions) > 1.000001):
        raise ValueError("Expected finite policy actions already bounded to [-1, 1]")
    centered = actions - actions.mean(dim=1, keepdim=True)
    scale = centered.abs().amax(dim=1, keepdim=True).clamp_min(1.0).reciprocal()
    return centered * scale, scale
