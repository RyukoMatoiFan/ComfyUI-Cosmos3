"""
cosmos3/loader.py

Model loading utilities for Cosmos3 Omni Nano:
  - load_cosmos3_model: loads 7 transformer shards -> ModelPatcher
  - load_cosmos3_vae: converts diffusers AutoencoderKLWan keys -> comfy VAE
  - load_cosmos3_tokenizer: loads Qwen2 tokenizer -> Cosmos3TextEncoderWrapper
"""

import json
import logging
import os
from typing import Dict

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.utils

from .model_base import Cosmos3ModelConfig, Cosmos3Omni
from .tokenizer import Cosmos3TextEncoderWrapper


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP8 helper
# ---------------------------------------------------------------------------

# Parameter name patterns that should NOT be cast to fp8 (kept in bf16).
# Follows comfy/sd.py load_diffusion_model_state_dict conventions.
_FP8_SKIP_PATTERNS = (
    "norm",
    "embed_tokens",
    "time_embedder",
    "bias",
)


def _should_skip_fp8(key: str, tensor: torch.Tensor) -> bool:
    """Return True if this tensor should stay in bf16 (not cast to fp8)."""
    if tensor.ndim <= 1:
        return True
    key_lower = key.lower()
    for pat in _FP8_SKIP_PATTERNS:
        if pat in key_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Key prefixes to drop from transformer state dict
# ---------------------------------------------------------------------------

# Drop: lm_head.*, action_* (the action head is not used by this port)
# KEEP: audio_proj_in.*, audio_proj_out.*, audio_modality_embed (used by the audio branch)
_FILTER_PREFIXES = ("lm_head.", "action_")
# Also drop rotary inv_freq buffers
_FILTER_SUFFIXES = ("rotary_emb.inv_freq",)


def _filter_transformer_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove keys that should not be loaded into Cosmos3OmniTransformer."""
    out = {}
    for k, v in sd.items():
        skip = False
        for prefix in _FILTER_PREFIXES:
            if k.startswith(prefix):
                skip = True
                break
        if not skip:
            for suffix in _FILTER_SUFFIXES:
                if k.endswith(suffix):
                    skip = True
                    break
        if not skip:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Checkpoint config -> transformer kwargs
# ---------------------------------------------------------------------------

# config.json keys the Cosmos3OmniTransformer constructor accepts verbatim.
# "dtype" is deliberately excluded: it is a string in the checkpoint config,
# while ComfyUI puts the torch dtype under that key via set_inference_dtype.
_CONFIG_PASSTHROUGH = (
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "intermediate_size",
    "latent_channel", "latent_patch_size", "vocab_size", "rope_theta",
    "rope_scaling", "base_fps", "enable_fps_modulation", "timestep_scale",
    "rms_norm_eps", "patch_latent_dim", "sound_gen", "sound_dim",
)

# Keys whose checkpoint name differs from the constructor parameter name.
_CONFIG_KEY_ALIASES = {
    "unified_3d_mrope_temporal_modality_margin": "mrope_margin",
}


def _load_transformer_config(transformer_dir: str) -> Dict:
    """Read transformer/config.json into Cosmos3OmniTransformer kwargs.

    Lets one loader cover checkpoints with different widths, depths and
    modalities (e.g. Nano generates sound, Super does not).
    """
    path = os.path.join(transformer_dir, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = {k: raw[k] for k in _CONFIG_PASSTHROUGH if k in raw}
    for src, dst in _CONFIG_KEY_ALIASES.items():
        if src in raw:
            cfg[dst] = raw[src]
    return cfg


# Shard index filenames, in the order they are probed.
_INDEX_FILENAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)
_SINGLE_FILENAMES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
)


def _resolve_shard_files(transformer_dir: str):
    """Return the transformer's safetensors files, sharded or single-file."""
    for name in _INDEX_FILENAMES:
        index_path = os.path.join(transformer_dir, name)
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shards = sorted(set(index["weight_map"].values()))
            return [os.path.join(transformer_dir, s) for s in shards]

    for name in _SINGLE_FILENAMES:
        single = os.path.join(transformer_dir, name)
        if os.path.exists(single):
            return [single]

    raise FileNotFoundError(
        f"No transformer weights found in {transformer_dir}. Expected one of "
        f"{_INDEX_FILENAMES} or {_SINGLE_FILENAMES}."
    )


# ---------------------------------------------------------------------------
# load_cosmos3_model
# ---------------------------------------------------------------------------

def load_cosmos3_model(
    model_dir: str,
    weight_dtype: str = "default",
) -> comfy.model_patcher.ModelPatcher:
    """
    Load a Cosmos3 Omni transformer from model_dir.

    The architecture is taken from transformer/config.json, so this covers both
    Cosmos3-Nano and the larger Cosmos3-Super checkpoints.

    Args:
        model_dir: Path to the checkpoint root (absolute path, or a bare folder
                   name resolved under ComfyUI/models/cosmos3/).
        weight_dtype: "default" (keep bf16) or "fp8_e4m3fn" (cast linear weights to fp8).

    Returns:
        comfy.model_patcher.ModelPatcher wrapping a Cosmos3Omni model.
    """
    transformer_dir = os.path.join(model_dir, "transformer")

    unet_config = _load_transformer_config(transformer_dir)
    logger.info(
        "Transformer config: hidden_size=%s layers=%s heads=%s sound_gen=%s",
        unet_config.get("hidden_size"), unet_config.get("num_hidden_layers"),
        unet_config.get("num_attention_heads"), unet_config.get("sound_gen", True),
    )

    shard_paths = _resolve_shard_files(transformer_dir)
    logger.info("Found %d transformer shard(s)", len(shard_paths))

    if weight_dtype == "fp8_e4m3fn":
        unet_dtype = torch.float8_e4m3fn
        manual_cast_dtype = torch.bfloat16
    else:
        unet_dtype = torch.bfloat16
        manual_cast_dtype = None

    model_config = Cosmos3ModelConfig(unet_config)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    offload_device = comfy.model_management.unet_offload_device()
    load_device = comfy.model_management.get_torch_device()

    logger.info("Creating Cosmos3Omni model on %s...", offload_device)
    model = Cosmos3Omni(model_config, device=offload_device)
    target = model.diffusion_model

    # Copy one shard at a time and release it, so only one shard is held in
    # memory alongside the model.
    expected = set(target.state_dict().keys())
    loaded = set()
    unexpected_all = []
    fp8_count = 0

    for shard_path in shard_paths:
        logger.info("  Loading %s", os.path.basename(shard_path))
        shard = comfy.utils.load_torch_file(shard_path, safe_load=True)
        shard = _filter_transformer_keys(shard)

        if weight_dtype == "fp8_e4m3fn":
            for k in list(shard.keys()):
                t = shard[k]
                if not _should_skip_fp8(k, t):
                    shard[k] = t.to(torch.float8_e4m3fn)
                    fp8_count += 1

        shard = model_config.process_unet_state_dict(shard)
        _, unexpected = target.load_state_dict(shard, strict=False)
        loaded.update(k for k in shard if k in expected)
        unexpected_all.extend(unexpected)
        del shard

    if weight_dtype == "fp8_e4m3fn":
        logger.info("Cast %d weights to fp8_e4m3fn", fp8_count)

    missing = sorted(expected - loaded)
    if missing:
        logger.error("MISSING keys in transformer (%d): %s", len(missing), missing[:20])
        raise RuntimeError(
            f"Transformer state dict has {len(missing)} missing keys! "
            f"First few: {missing[:5]}"
        )

    if unexpected_all:
        logger.warning(
            "Unexpected keys in transformer (%d): %s",
            len(unexpected_all),
            unexpected_all[:10],
        )

    logger.info("Transformer loaded successfully. Wrapping in ModelPatcher...")

    # CoreModelPatcher, not ModelPatcher: main.py rebinds it to ModelPatcherDynamic
    # at startup when comfy-aimdo is available, which is what gives us dynamic VRAM.
    # Looked up on the module at call time so the startup rebind is picked up;
    # falls back to ModelPatcher on ComfyUI versions predating the dynamic path.
    patcher_cls = getattr(
        comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher
    )
    patcher = patcher_cls(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )

    return patcher


# ---------------------------------------------------------------------------
# VAE key conversion: diffusers AutoencoderKLWan -> comfy WanVAE (vae2_2)
# ---------------------------------------------------------------------------
#
# Diffusers AutoencoderKLWan module layout:
#   encoder.conv_in                -> encoder.conv1
#   encoder.conv_out               -> encoder.head.2
#   encoder.norm_out               -> encoder.head.0
#   encoder.down_blocks.{i}        -> encoder.downsamples.{i}  (Down_ResidualBlock)
#     .resnets.{j}                 -> .downsamples.{j}  (ResidualBlock inside downsamples)
#       .conv1                     -> .residual.2
#       .conv2                     -> .residual.6
#       .norm1.gamma               -> .residual.0.gamma
#       .norm2.gamma               -> .residual.3.gamma
#       .conv_shortcut             -> .shortcut
#     .downsampler                 -> .downsamples.{num_res_blocks}  (Resample)
#       .resample.1                -> .resample.1
#       .time_conv                 -> .time_conv
#   encoder.mid_block              -> encoder.middle
#     .resnets.{0,1}              -> .{0,2}  (ResidualBlocks, skipping attention at .1)
#       same resnet key mapping
#     .attentions.0               -> .1  (AttentionBlock)
#       .norm.gamma               -> .norm.gamma
#       .to_qkv                   -> .to_qkv
#       .proj                     -> .proj
#
#   decoder.conv_in                -> decoder.conv1
#   decoder.conv_out               -> decoder.head.2
#   decoder.norm_out               -> decoder.head.0
#   decoder.up_blocks.{i}          -> decoder.upsamples.{i}  (Up_ResidualBlock)
#     .resnets.{0}                 -> .upsamples.{0}  (first resnet has shortcut if channel change)
#       same resnet mapping
#     .upsampler                   -> .upsamples.{num_res_blocks+1}  (Resample, last in upsamples)
#   decoder.mid_block              -> decoder.middle (same as encoder.mid_block)
#
#   quant_conv                     -> conv1  (WanVAE top-level)
#   post_quant_conv                -> conv2
#
# IMPORTANT: diffusers resnets inside down/up blocks are numbered 0..N-1,
# but inside a Down/Up_ResidualBlock's self.downsamples/upsamples Sequential,
# there are (num_res_blocks) ResidualBlock entries followed by a Resample.
# The diffusers mid_block has resnets=[0,1] and attentions=[0]:
#   layout: resnet0, attention0, resnet1
#   -> middle[0], middle[1], middle[2]
#
# ResidualBlock.residual is a Sequential:
#   [0] RMS_norm   -> .gamma   (no "weight" key, just "gamma")
#   [1] SiLU
#   [2] CausalConv3d  -> conv1
#   [3] RMS_norm   -> .gamma
#   [4] SiLU
#   [5] Dropout
#   [6] CausalConv3d  -> conv2
#
# AttentionBlock (encoder/decoder mid):
#   norm.gamma, to_qkv.weight/.bias, proj.weight/.bias


def _convert_vae_state_dict(diffusers_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert diffusers AutoencoderKLWan state dict to comfy WanVAE (vae2_2) format.

    Returns the converted state dict.
    """
    out = {}

    def _resnet_key(diffusers_sub: str) -> str:
        """
        Convert diffusers resnet sub-key to comfy residual key.
        diffusers_sub: e.g. "conv1.weight", "norm1.gamma", "conv_shortcut.weight"
        returns comfy residual path suffix.
        """
        if diffusers_sub == "conv1.weight":
            return "residual.2.weight"
        elif diffusers_sub == "conv1.bias":
            return "residual.2.bias"
        elif diffusers_sub == "conv2.weight":
            return "residual.6.weight"
        elif diffusers_sub == "conv2.bias":
            return "residual.6.bias"
        elif diffusers_sub == "norm1.gamma":
            return "residual.0.gamma"
        elif diffusers_sub == "norm2.gamma":
            return "residual.3.gamma"
        elif diffusers_sub == "conv_shortcut.weight":
            return "shortcut.weight"
        elif diffusers_sub == "conv_shortcut.bias":
            return "shortcut.bias"
        else:
            raise ValueError(f"Unknown resnet sub-key: {diffusers_sub!r}")

    def _attn_key(diffusers_sub: str) -> str:
        """Convert diffusers attention sub-key to comfy AttentionBlock key."""
        # diffusers: norm.gamma, to_qkv.weight, to_qkv.bias, proj.weight, proj.bias
        return diffusers_sub  # same naming in comfy AttentionBlock

    # Process all keys
    for diff_key, tensor in diffusers_sd.items():
        parts = diff_key.split(".")

        # ----------------------------------------------------------------
        # quant_conv -> conv1 (WanVAE top level)
        # post_quant_conv -> conv2
        # ----------------------------------------------------------------
        if diff_key.startswith("quant_conv."):
            suffix = diff_key[len("quant_conv."):]
            comfy_key = f"conv1.{suffix}"
            out[comfy_key] = tensor
            continue

        if diff_key.startswith("post_quant_conv."):
            suffix = diff_key[len("post_quant_conv."):]
            comfy_key = f"conv2.{suffix}"
            out[comfy_key] = tensor
            continue

        # ----------------------------------------------------------------
        # encoder.*
        # ----------------------------------------------------------------
        if parts[0] == "encoder":
            rest = parts[1:]

            if rest[0] == "conv_in":
                # encoder.conv_in -> encoder.conv1
                out[f"encoder.conv1.{rest[1]}"] = tensor
                continue

            if rest[0] == "conv_out":
                # encoder.conv_out -> encoder.head.2
                out[f"encoder.head.2.{rest[1]}"] = tensor
                continue

            if rest[0] == "norm_out":
                # encoder.norm_out.gamma -> encoder.head.0.gamma
                out[f"encoder.head.0.{rest[1]}"] = tensor
                continue

            if rest[0] == "mid_block":
                # encoder.mid_block.resnets.{0,1}.X or .attentions.0.X
                # comfy layout: middle[0], middle[1](attn), middle[2]
                if rest[1] == "resnets":
                    resnet_idx = int(rest[2])
                    sub = ".".join(rest[3:])
                    # resnets.0 -> middle.0, resnets.1 -> middle.2
                    comfy_idx = 0 if resnet_idx == 0 else 2
                    rk = _resnet_key(sub)
                    out[f"encoder.middle.{comfy_idx}.{rk}"] = tensor
                    continue
                elif rest[1] == "attentions":
                    # attentions.0 -> middle.1
                    sub = ".".join(rest[3:])
                    ck = _attn_key(sub)
                    out[f"encoder.middle.1.{ck}"] = tensor
                    continue

            if rest[0] == "down_blocks":
                block_idx = int(rest[1])
                rest2 = rest[2:]

                if rest2[0] == "resnets":
                    resnet_idx = int(rest2[1])
                    sub = ".".join(rest2[2:])
                    rk = _resnet_key(sub)
                    out[f"encoder.downsamples.{block_idx}.downsamples.{resnet_idx}.{rk}"] = tensor
                    continue

                if rest2[0] == "downsampler":
                    # .downsampler.resample.1.* or .downsampler.time_conv.*
                    sub = ".".join(rest2[1:])
                    # num_res_blocks = 2, so downsampler is at index 2
                    out[f"encoder.downsamples.{block_idx}.downsamples.2.{sub}"] = tensor
                    continue

            logger.warning("Unmatched encoder key: %s", diff_key)
            out[diff_key] = tensor
            continue

        # ----------------------------------------------------------------
        # decoder.*
        # ----------------------------------------------------------------
        if parts[0] == "decoder":
            rest = parts[1:]

            if rest[0] == "conv_in":
                out[f"decoder.conv1.{rest[1]}"] = tensor
                continue

            if rest[0] == "conv_out":
                out[f"decoder.head.2.{rest[1]}"] = tensor
                continue

            if rest[0] == "norm_out":
                out[f"decoder.head.0.{rest[1]}"] = tensor
                continue

            if rest[0] == "mid_block":
                if rest[1] == "resnets":
                    resnet_idx = int(rest[2])
                    sub = ".".join(rest[3:])
                    comfy_idx = 0 if resnet_idx == 0 else 2
                    rk = _resnet_key(sub)
                    out[f"decoder.middle.{comfy_idx}.{rk}"] = tensor
                    continue
                elif rest[1] == "attentions":
                    sub = ".".join(rest[3:])
                    ck = _attn_key(sub)
                    out[f"decoder.middle.1.{ck}"] = tensor
                    continue

            if rest[0] == "up_blocks":
                block_idx = int(rest[1])
                rest2 = rest[2:]

                if rest2[0] == "resnets":
                    resnet_idx = int(rest2[1])
                    sub = ".".join(rest2[2:])
                    rk = _resnet_key(sub)
                    out[f"decoder.upsamples.{block_idx}.upsamples.{resnet_idx}.{rk}"] = tensor
                    continue

                if rest2[0] == "upsampler":
                    sub = ".".join(rest2[1:])
                    # decoder Up_ResidualBlock: (num_res_blocks+1)=3 residual blocks + Resample
                    # Resample is at index 3 in upsamples Sequential
                    out[f"decoder.upsamples.{block_idx}.upsamples.3.{sub}"] = tensor
                    continue

            logger.warning("Unmatched decoder key: %s", diff_key)
            out[diff_key] = tensor
            continue

        # Unmatched
        logger.warning("Unmatched VAE key: %s", diff_key)
        out[diff_key] = tensor

    return out


# ---------------------------------------------------------------------------
# load_cosmos3_vae
# ---------------------------------------------------------------------------

def load_cosmos3_vae(model_dir: str) -> comfy.sd.VAE:
    """
    Load the Cosmos3 VAE (AutoencoderKLWan = Wan2.2 VAE) from model_dir.

    Converts diffusers key format to comfy WanVAE (vae2_2) format.
    Asserts exact key-set match after loading.

    Returns:
        comfy.sd.VAE instance.
    """
    vae_path = os.path.join(model_dir, "vae", "diffusion_pytorch_model.safetensors")
    logger.info("Loading VAE from %s", vae_path)

    diffusers_sd = comfy.utils.load_torch_file(vae_path, safe_load=True)
    logger.info("Loaded %d raw VAE keys", len(diffusers_sd))

    # Convert keys
    converted_sd = _convert_vae_state_dict(diffusers_sd)
    logger.info("Converted to %d comfy VAE keys", len(converted_sd))

    del diffusers_sd

    # Build VAE via comfy.sd.VAE
    vae = comfy.sd.VAE(sd=converted_sd)

    # Verify exact key-set match
    model_keys = set(vae.first_stage_model.state_dict().keys())
    converted_keys = set(converted_sd.keys())

    missing_in_model = converted_keys - model_keys
    missing_in_converted = model_keys - converted_keys

    if missing_in_model or missing_in_converted:
        logger.error(
            "VAE key mismatch!\n"
            "  Keys in converted but NOT in model (%d): %s\n"
            "  Keys in model but NOT in converted (%d): %s",
            len(missing_in_model),
            sorted(missing_in_model)[:10],
            len(missing_in_converted),
            sorted(missing_in_converted)[:10],
        )
        raise RuntimeError(
            f"VAE key conversion failed: "
            f"{len(missing_in_model)} extra keys, "
            f"{len(missing_in_converted)} missing keys"
        )

    logger.info("VAE loaded successfully: %d keys, exact match.", len(model_keys))
    return vae


# ---------------------------------------------------------------------------
# load_cosmos3_tokenizer
# ---------------------------------------------------------------------------

def load_cosmos3_tokenizer(model_dir: str) -> Cosmos3TextEncoderWrapper:
    """
    Load the Cosmos3 text tokenizer (Qwen2TokenizerFast).

    Args:
        model_dir: Path to the cosmos3-nano checkpoint root.

    Returns:
        Cosmos3TextEncoderWrapper instance.
    """
    logger.info("Loading Cosmos3 tokenizer from %s/text_tokenizer", model_dir)
    return Cosmos3TextEncoderWrapper(model_dir)
