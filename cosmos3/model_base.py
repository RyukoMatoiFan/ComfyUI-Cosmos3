"""
cosmos3/model_base.py

BaseModel subclass and model config for Cosmos3 Omni Nano.

  - Cosmos3ModelSampling: identity timestep mapping over a shifted flow schedule.
  - Cosmos3Omni(BaseModel): extra_conds for cross-attn, fps, I2V latent/mask, sound.
  - Cosmos3ModelConfig(BASE): unet_config={}, latent_format=Wan22.

_apply_model keeps the conditioning (token IDs) in float32 rather than the model
dtype: bf16 cannot represent integer IDs above 256, so a bf16 cast would corrupt
the Qwen2 vocabulary (up to 151936).
"""

import logging

import torch

import comfy.model_base
import comfy.model_management
import comfy.model_sampling
import comfy.conds
import comfy.latent_formats
import comfy.supported_models_base
import comfy.ops
from comfy.model_base import convert_tensor
import comfy.utils as utils

from .av_pack import (
    extra_frames as _extra_frames,
    region_to_sound_raw,
    sound_raw_to_region_velocity,
)

def _get_transformer_class():
    from .model import Cosmos3OmniTransformer
    return Cosmos3OmniTransformer


# ---------------------------------------------------------------------------
# Cosmos3ModelSampling
# ---------------------------------------------------------------------------

class Cosmos3ModelSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    """
    Model sampling for Cosmos3: identity timestep/sigma mapping over a uniform
    flow schedule with a discrete (SD3/Flux-style) shift — matching the NVIDIA
    cosmos-framework FlowUniPCMultistepScheduler.

    The model receives plain comfy sigma s ∈ (0,1] unchanged; inside
    Cosmos3OmniTransformer.forward it is fed to the timestep embedding directly
    (reference: scheduler.timesteps = s*1000, then *timestep_scale 1e-3 -> s).

    The sigma table is `shift * t / (1 + (shift-1)*t)` for t in (0,1], so plain
    KSampler schedulers ("normal"/"simple"/etc.) also produce shifted sigmas.
    For exact control use the Cosmos3 Scheduler node + SamplerCustomAdvanced.
    """

    _DEFAULT_SHIFT = 10.0

    def set_parameters(self, shift=10.0, timesteps=1000, multiplier=1.0):
        """Build a 1000-point uniform-flow sigma table with the discrete shift."""
        self.shift = float(shift) if shift else self._DEFAULT_SHIFT
        self.multiplier = 1.0  # FORCE identity; never rescale sigma

        # t in (0, 1]; s = shift*t / (1 + (shift-1)*t). Ascending so sigmas[-1] ~= 1.
        t = torch.linspace(1.0 / timesteps, 1.0, timesteps)
        s = self.shift * t / (1.0 + (self.shift - 1.0) * t)
        self.register_buffer("sigmas", s)

    # Identity mapping: timestep == sigma (both in [0,1])
    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return timestep

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        t = 1.0 - percent  # percent 0 -> high noise
        return self.shift * t / (1.0 + (self.shift - 1.0) * t)


# ---------------------------------------------------------------------------
# Cosmos3ModelConfig
# ---------------------------------------------------------------------------

class Cosmos3ModelConfig(comfy.supported_models_base.BASE):
    """
    Model config for Cosmos3 Omni Nano.

    unet_config is empty {} — we create the model directly via get_model(),
    not via ComfyUI's automatic detection machinery.
    """

    # Empty unet_config: not matched against checkpoint keys; loader calls get_model() directly.
    unet_config = {}
    unet_extra_config = {}

    # Use Wan 2.2 latent format (48-channel, 16x spatial, 4x temporal)
    latent_format = comfy.latent_formats.Wan22

    # Sampling settings: multiplier=1 for identity; shift unused by our custom class
    sampling_settings = {
        "multiplier": 1.0,
        "shift": 10.0,
    }

    # Supported dtypes: bf16 native, fp32 fallback
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    # Memory: 16B joint-attention model, slightly heavier than WAN
    # WAN21_T2V uses 0.9; Cosmos3 is roughly 18x heavier → keep at 1.8
    memory_usage_factor = 1.8

    # No clip/vae prefix (custom loader handles everything)
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        """Return a Cosmos3Omni model instance."""
        return Cosmos3Omni(self, device=device)

    def clip_target(self, state_dict={}):
        """No CLIP text encoder for Cosmos3."""
        return None


# ---------------------------------------------------------------------------
# Cosmos3Omni
# ---------------------------------------------------------------------------

class Cosmos3Omni(comfy.model_base.BaseModel):
    """
    ComfyUI BaseModel subclass for Cosmos3 Omni Nano.

    The model receives token IDs (not embeddings) as `context`, fps as a scalar,
    and optional I2V conditioning tensors.
    """

    def __init__(
        self,
        model_config: Cosmos3ModelConfig,
        model_type=comfy.model_base.ModelType.FLOW,
        device=None,
    ):
        super().__init__(
            model_config,
            model_type,
            device=device,
            unet_model=_get_transformer_class(),
        )

        # Override model_sampling with our identity-mapping version
        # (the base __init__ already called model_sampling() which created a
        # ModelSamplingDiscreteFlow instance; we replace it with ours)
        self.model_sampling = Cosmos3ModelSampling(model_config)

        logging.info(
            "Cosmos3Omni: sigma_min=%.4f sigma_max=%.4f",
            self.model_sampling.sigma_min.item(),
            self.model_sampling.sigma_max.item(),
        )

    def _apply_model(
        self, x, t, c_concat=None, c_crossattn=None, control=None,
        transformer_options={}, **kwargs
    ):
        """
        Like BaseModel._apply_model, with two differences:
          1. context (token IDs) is moved to device but kept float32 (see class docstring).
          2. with `cosmos3_sound_tokens`, the packed AV latent is split into video +
             sound, the model is called with both, and the velocities are re-merged.
        """
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat(
                [xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)],
                dim=1,
            )

        context = c_crossattn
        dtype = self.get_dtype_inference()

        xc = xc.to(dtype)
        device = xc.device
        t = self.model_sampling.timestep(t).float()

        # Move context to device but keep float32 (token IDs; bf16 corrupts IDs > 256).
        if context is not None:
            context = comfy.model_management.cast_to_device(context, device, None)

        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                extra = convert_tensor(extra, dtype, device)
            elif isinstance(extra, list):
                ex = []
                for ext in extra:
                    ex.append(convert_tensor(ext, dtype, device))
                extra = ex
            extra_conds[o] = extra

        t_proc = self.process_timestep(t, x=x, **extra_conds)
        if "latent_shapes" in extra_conds:
            xc = utils.unpack_latents(xc, extra_conds.pop("latent_shapes"))

        transformer_options = transformer_options.copy()
        transformer_options["prefetch_dynamic_vbars"] = (
            self.current_patcher is not None and self.current_patcher.is_dynamic()
        )

        # ── AV path: split packed latent, call model with sound, merge back ──
        t_sound = extra_conds.pop("cosmos3_sound_tokens", None)
        if isinstance(t_sound, torch.Tensor):
            t_sound = t_sound.item()
        if isinstance(t_sound, (int, float)) and int(t_sound) > 0:
            t_sound = int(t_sound)
            B, C, T_total, H, W = xc.shape  # (B, 48, Tv+n_extra, H, W)
            n_extra = _extra_frames(t_sound, H, W)
            t_video = T_total - n_extra

            # Split into video and packed tail
            xc_video = xc[:, :, :t_video, :, :]   # (B, 48, Tv, H, W)
            tail_region = xc[:, :, t_video:, :, :] # (B, 48, n_extra, H, W)

            # Obtain Wan22 latent affine tensors (on correct device/dtype)
            lf = self.latent_format
            latents_mean = lf.latents_mean.to(dtype=dtype, device=device)
            latents_std  = lf.latents_std.to(dtype=dtype, device=device)

            # Un-normalise packed tail → extract raw sound latent (B, 64, T_s)
            sound_raw = region_to_sound_raw(
                tail_region, latents_mean, latents_std, t_sound
            )

            # Call diffusion model with x_video; it returns (v_vision, v_sound_raw)
            model_out = self.diffusion_model(
                xc_video, t_proc, context=context, control=control,
                transformer_options=transformer_options,
                cosmos3_sound_latent=sound_raw,
                **extra_conds,
            )

            # Unpack tuple output
            assert isinstance(model_out, tuple) and len(model_out) == 2, (
                f"Expected (v_vision, v_sound) tuple from diffusion_model when "
                f"cosmos3_sound_latent provided, got: {type(model_out)}"
            )
            v_vision, v_sound_raw = model_out  # (B,48,Tv,H,W), (B,64,T_s)

            # Scatter v_sound_raw back into packed tail velocity (divide by std)
            v_tail = sound_raw_to_region_velocity(
                v_sound_raw, latents_std, n_extra, H, W
            )

            # Concatenate to form full packed velocity (B, 48, Tv+n_extra, H, W)
            model_output = torch.cat([v_vision, v_tail], dim=2)

        else:
            # Video-only path
            model_output = self.diffusion_model(
                xc, t_proc, context=context, control=control,
                transformer_options=transformer_options, **extra_conds,
            )
            if len(model_output) > 1 and not torch.is_tensor(model_output):
                model_output, _ = utils.pack_latents(model_output)

        return self.model_sampling.calculate_denoised(sigma, model_output.float(), x)

    # ------------------------------------------------------------------
    # extra_conds
    # ------------------------------------------------------------------

    def extra_conds(self, **kwargs):
        """
        Build conditioning dict passed to Cosmos3OmniTransformer.forward().

        Keys produced:
          c_crossattn   : CONDRegular — token IDs tensor (1, n_tokens, 1) float32
          cosmos3_fps   : CONDConstant — frames per second (float)
          cosmos3_cond_latent : CONDRegular — normalized I2V cond latent (optional)
          cosmos3_cond_mask   : CONDRegular — I2V mask (optional)
        """
        out = super().extra_conds(**kwargs)

        # -- cross-attention: token IDs from conditioning dict --
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        # -- FPS conditioning --
        fps = kwargs.get("cosmos3_fps", None)
        if fps is None:
            fps = 24.0
        out["cosmos3_fps"] = comfy.conds.CONDConstant(fps)

        # -- I2V: conditioned latent (normalize then wrap) --
        cond_latent = kwargs.get("cosmos3_cond_latent", None)
        if cond_latent is not None:
            # Apply same normalization as model input (process_in uses (x - mean) / std)
            cond_latent_norm = self.latent_format.process_in(cond_latent)
            out["cosmos3_cond_latent"] = comfy.conds.CONDRegular(cond_latent_norm)

        # -- I2V: per-frame mask --
        cond_mask = kwargs.get("cosmos3_cond_mask", None)
        if cond_mask is not None:
            out["cosmos3_cond_mask"] = comfy.conds.CONDRegular(cond_mask)

        # -- AV: sound token count (positive int signals AV mode) --
        sound_tokens = kwargs.get("cosmos3_sound_tokens", None)
        if sound_tokens is not None and int(sound_tokens) > 0:
            out["cosmos3_sound_tokens"] = comfy.conds.CONDConstant(int(sound_tokens))

        return out

    def memory_required(self, input_shape, cond_shapes={}):
        """
        Estimate GPU memory for lowvram budget.

        Joint attention model: sequence length ≈ text_tokens + T*ceil(H/2)*ceil(W/2).
        For default 189 frames @ 720p: seq_len ≈ 200 + 48*23*40 ≈ 44360 tokens.
        Scale ~proportional to sequence^2 for attention.

        We reuse BaseModel's formula but scale by our memory_usage_factor.
        """
        # Use base implementation (formula from BaseModel.memory_required)
        # but our memory_usage_factor=1.8 is already set
        return super().memory_required(input_shape, cond_shapes)
