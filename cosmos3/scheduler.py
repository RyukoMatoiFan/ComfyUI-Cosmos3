"""
cosmos3/scheduler.py
Compute Cosmos3 flow-matching sigmas in ComfyUI (comfy-sigma) format.

This matches the NVIDIA cosmos-framework FlowUniPCMultistepScheduler used for
Cosmos3 inference: a uniform flow schedule with a discrete (SD3/Flux-style) shift.

    sigmas = linspace(1, 0, steps + 1)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)      # if shift != 1
    timesteps = sigmas * 1000                                 # (the model side handles this)

`shift` (discrete_flow_shift / flow_shift) defaults to 10.0 and is
resolution-dependent: 10 @ 720p, 5 @ 480p, 3 @ 256p.
"""

import numpy as np
import torch


def _apply_flow_shift(sigmas: np.ndarray, shift: float) -> np.ndarray:
    """Discrete flow shift: s' = shift * s / (1 + (shift - 1) * s). Endpoints 0,1 are fixed."""
    if abs(shift - 1.0) < 1e-9:
        return sigmas
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def compute_cosmos3_sigmas(
    steps: int,
    flow_shift: float = 10.0,
    denoise: float = 1.0,
) -> torch.FloatTensor:
    """
    Cosmos3 flow-matching sigmas for ComfyUI samplers.

    Args:
        steps: number of denoising steps.
        flow_shift: discrete flow shift (10 @ 720p, 5 @ 480p, 3 @ 256p; 1 = no shift).
        denoise: denoising strength in (0, 1]. < 1 computes a longer schedule and
                 keeps its tail (the ComfyUI BasicScheduler convention).

    Returns:
        1-D FloatTensor of length steps+1, descending from ~1.0 to 0.0 (comfy sigmas).
    """
    if denoise <= 0.0:
        raise ValueError("denoise must be > 0")

    total_steps = int(round(steps / denoise)) if denoise < 1.0 else steps

    # Uniform flow schedule over [1, 0], then discrete shift. Endpoint 0 is preserved.
    sigmas = np.linspace(1.0, 0.0, total_steps + 1, dtype=np.float64)
    sigmas = _apply_flow_shift(sigmas, float(flow_shift))

    if denoise < 1.0:
        # keep the last `steps` non-zero sigmas + the trailing 0
        sigmas = sigmas[total_steps - steps:]

    return torch.from_numpy(sigmas.astype(np.float32))
