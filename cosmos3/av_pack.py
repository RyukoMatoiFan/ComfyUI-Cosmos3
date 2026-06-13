"""
cosmos3/av_pack.py

Latent-packing helpers for joint audio-video generation.

Sound latent (B, 64, T_s) is packed into extra latent frames appended along
the T axis of the video latent (B, 48, T_video_lat, H_lat, W_lat).  The full
packed tensor is a standard Wan22 latent that ComfyUI samplers can denoise
without any modification.

Packing layout (C-contiguous in (C, T, H, W) order):
  - Extra frames region: (B, 48, n_extra, H, W)  flattened → (B, 48*n_extra*H*W)
  - First 64*T_s elements store the sound latent flattened as (64, T_s) row-major.
  - Remaining elements are zero padding.

All affine helpers (un-normalise, velocity scatter) are pure-torch and accept the
Wan22 latents_mean / latents_std tensors as arguments so this module has no comfy
imports and is unit-testable in isolation.
"""

import math
import torch

# ── Constants ────────────────────────────────────────────────────────────────

SOUND_DIM   = 64      # latent channels for audio (audio_proj_in/out dim)
HOP         = 1920    # AVAE hop size = product of decoder strides (2*4*5*6*8)
SAMPLE_RATE = 48000

# ── Size arithmetic ──────────────────────────────────────────────────────────

def sound_token_count(num_frames: int, fps: float) -> int:
    """Number of sound latent tokens for *num_frames* video frames at *fps*.

    T_sound = ceil(audio_samples / HOP)
    """
    n_audio_samples = int(num_frames / fps * SAMPLE_RATE)
    return math.ceil(n_audio_samples / HOP)


def audio_sample_count(num_frames: int, fps: float) -> int:
    """Exact waveform length (samples) for *num_frames* video frames at *fps*."""
    return int(num_frames / fps * SAMPLE_RATE)


def extra_frames(t_sound: int, h_lat: int, w_lat: int) -> int:
    """Number of extra latent frames required to hold *t_sound* sound tokens.

    n_extra = ceil(SOUND_DIM * T_sound / (48 * H_lat * W_lat))
    Always >= 1 when t_sound > 0.
    """
    return math.ceil((SOUND_DIM * t_sound) / (48 * h_lat * w_lat))


# ── Pack / unpack ────────────────────────────────────────────────────────────

def pack_sound(
    video_lat: torch.Tensor,   # (B, 48, Tv, H, W)
    sound_lat: torch.Tensor,   # (B, 64, Ts)
) -> torch.Tensor:
    """Pack *sound_lat* into extra frames appended to *video_lat*.

    Returns packed tensor (B, 48, Tv + n_extra, H, W).

    Layout of the extra-frames region (flattened C-contiguous over C,T,H,W):
      positions [0 .. 64*Ts - 1] = sound_lat.reshape(B, -1) (channel-major)
      positions [64*Ts ..]        = 0 (padding)
    """
    B, C, Tv, H, W = video_lat.shape
    assert C == 48, f"Expected 48-channel video latent, got C={C}"
    Ts = sound_lat.shape[2]
    assert sound_lat.shape == (B, SOUND_DIM, Ts), \
        f"Expected sound_lat shape (B,{SOUND_DIM},Ts), got {sound_lat.shape}"

    n_extra = extra_frames(Ts, H, W)
    tail_numel = 48 * n_extra * H * W          # per-batch total elements in tail

    # Build flat tail (B, tail_numel) initialised to zeros
    tail_flat = video_lat.new_zeros(B, tail_numel)

    # Sound latent flattened channel-major: (B, 64*Ts)
    sound_flat = sound_lat.reshape(B, SOUND_DIM * Ts)    # row-major (channel-major)
    assert SOUND_DIM * Ts <= tail_numel, \
        f"Sound latent ({SOUND_DIM*Ts}) exceeds tail capacity ({tail_numel})"

    tail_flat[:, : SOUND_DIM * Ts] = sound_flat

    # Reshape tail back to (B, 48, n_extra, H, W) — C-contiguous view
    tail = tail_flat.view(B, 48, n_extra, H, W)

    return torch.cat([video_lat, tail], dim=2)   # (B, 48, Tv+n_extra, H, W)


def unpack_sound(
    packed: torch.Tensor,    # (B, 48, Tv+n_extra, H, W)
    t_video_lat: int,
    t_sound: int,
) -> tuple:
    """Inverse of *pack_sound*.

    Returns (video_lat (B,48,Tv,H,W), sound_lat (B,64,Ts)).
    """
    B, C, T_total, H, W = packed.shape
    assert C == 48
    n_extra = extra_frames(t_sound, H, W)
    assert T_total == t_video_lat + n_extra, \
        f"T_total={T_total} != t_video_lat({t_video_lat}) + n_extra({n_extra})"

    video_lat = packed[:, :, :t_video_lat, :, :]

    # Extract tail and read sound elements
    tail = packed[:, :, t_video_lat:, :, :]                  # (B, 48, n_extra, H, W)
    tail_flat = tail.reshape(B, 48 * n_extra * H * W)        # (B, tail_numel)
    sound_flat = tail_flat[:, : SOUND_DIM * t_sound]          # (B, 64*Ts)
    sound_lat = sound_flat.reshape(B, SOUND_DIM, t_sound)     # (B, 64, Ts)

    return video_lat, sound_lat


# ── Affine helpers ───────────────────────────────────────────────────────────
# These are the ONLY places where the Wan22 latents_mean/std affine is applied
# to the sound region.  model_base.py calls these helpers exclusively.
#
# Wan22 process_in:  x_norm = (x_raw - mean) / std   (scale_factor=1)
# Wan22 process_out: x_raw  = x_norm * std + mean
#
# mean, std shapes are (1, 48, 1, 1, 1).  We reshape to (1, 48, 1, 1, 1) for
# broadcasting over (B, 48, n_extra, H, W).

def unnorm_packed_region(
    region: torch.Tensor,          # (B, 48, n_extra, H, W)  — normalised
    latents_mean: torch.Tensor,    # (1, 48, 1, 1, 1)
    latents_std: torch.Tensor,     # (1, 48, 1, 1, 1)
) -> torch.Tensor:
    """Un-normalise a packed tail region and extract the raw sound latent.

    Step 1: region_raw = region * std + mean     (invert process_in)
    Step 2: flatten C-contiguous → take first SOUND_DIM*T_s elements
    Step 3: reshape → (B, 64, T_s)
    """
    B, C, n_extra, H, W = region.shape
    mean = latents_mean.to(dtype=region.dtype, device=region.device)
    std  = latents_std.to(dtype=region.dtype, device=region.device)

    # Un-normalise (invert process_in: x_raw = x_norm * std + mean)
    region_raw = region * std + mean  # (B, 48, n_extra, H, W)

    # Flatten C-contiguous over (C, n_extra, H, W)
    tail_flat = region_raw.reshape(B, C * n_extra * H * W)

    # Determine T_s from available elements
    numel_sound = tail_flat.shape[1]  # upper bound
    # Caller must know T_s; we return the whole flat view and let caller slice
    return region_raw, tail_flat


def region_to_sound_raw(
    region: torch.Tensor,          # (B, 48, n_extra, H, W) — normalised
    latents_mean: torch.Tensor,    # (1, 48, 1, 1, 1)
    latents_std: torch.Tensor,     # (1, 48, 1, 1, 1)
    t_sound: int,
) -> torch.Tensor:
    """Un-normalise packed tail region and return raw sound latent (B, 64, T_s).

    Math:
        region_raw = region * std + mean
        sound_raw  = region_raw.flatten(C-contiguous)[:, :64*T_s].reshape(B,64,T_s)
    """
    B, C, n_extra, H, W = region.shape
    mean = latents_mean.to(dtype=region.dtype, device=region.device)
    std  = latents_std.to(dtype=region.dtype, device=region.device)

    region_raw = region * std + mean
    tail_flat  = region_raw.reshape(B, C * n_extra * H * W)
    sound_flat = tail_flat[:, : SOUND_DIM * t_sound]
    return sound_flat.reshape(B, SOUND_DIM, t_sound)


def sound_raw_to_region_velocity(
    v_sound_raw: torch.Tensor,     # (B, 64, T_s)
    latents_std: torch.Tensor,     # (1, 48, 1, 1, 1)
    n_extra: int,
    h_lat: int,
    w_lat: int,
) -> torch.Tensor:
    """Scatter v_sound_raw into a packed tail velocity region (B, 48, n_extra, H, W).

    The velocity in packed (normalised) space is:
        v_packed = v_raw / std     (NO mean term for velocities)

    Un-occupied elements (padding) are zero.
    """
    B, SD, Ts = v_sound_raw.shape
    assert SD == SOUND_DIM, f"Expected SOUND_DIM={SOUND_DIM}, got {SD}"

    std = latents_std.to(dtype=v_sound_raw.dtype, device=v_sound_raw.device)

    # Build flat tail (B, 48*n_extra*H*W) of zeros
    tail_numel = 48 * n_extra * h_lat * w_lat
    tail_flat  = v_sound_raw.new_zeros(B, tail_numel)

    # Scatter velocity (no mean): v_flat = v_sound_raw / std (applied after reshape)
    # We first build the raw packed tail, then divide by std channel-wise.
    v_flat = v_sound_raw.reshape(B, SOUND_DIM * Ts)  # (B, 64*Ts)
    tail_flat[:, : SOUND_DIM * Ts] = v_flat

    # Reshape to (B, 48, n_extra, H, W) for per-channel std division
    tail = tail_flat.view(B, 48, n_extra, h_lat, w_lat)

    # Divide by std (velocity transform: NO mean)
    tail = tail / std   # broadcast over (B, 48, n_extra, H, W)

    # Zero out padding positions in normalised space (they were zero in raw space
    # so raw/std could produce NaN if std=0; all Wan22 stds are positive, so fine)
    return tail
