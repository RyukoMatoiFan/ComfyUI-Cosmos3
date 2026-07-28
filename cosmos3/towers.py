"""
cosmos3/towers.py

Which checkpoint keys belong to which half of the model, and where a
checkpoint's shards live. Deliberately free of torch and comfy imports so the
offline splitter in tools/ can share exactly the classification the runtime
loader uses — the two disagreeing would produce a silently broken split.

Cosmos3 is a Mixture-of-Transformers: the understanding ("und", or reasoner)
pathway and the generation pathway run through the same layer stack but use
entirely disjoint weights. The und half is causal over the prompt alone, so its
per-layer K/V do not change across denoising steps: it runs once, before
sampling, and never has to be resident while the denoiser works.
"""

import json
import os


# ---------------------------------------------------------------------------
# Keys this port never loads
# ---------------------------------------------------------------------------

# lm_head is the LLM's vocabulary projection — only needed to emit text tokens,
# which video generation never does. action_* is the action head.
FILTER_PREFIXES = ("lm_head.", "action_")
# Rotary tables are recomputed, not loaded.
FILTER_SUFFIXES = ("rotary_emb.inv_freq",)


def is_dropped_key(key: str) -> bool:
    """True if this key is not loaded into Cosmos3OmniTransformer at all."""
    return key.startswith(FILTER_PREFIXES) or key.endswith(FILTER_SUFFIXES)


# ---------------------------------------------------------------------------
# und / gen classification
# ---------------------------------------------------------------------------

# Attention sub-keys, matched after ".self_attn.". k_norm_und_for_gen is applied
# during the prefill, so it belongs to the und side despite its name.
_UND_ATTN = ("to_q.", "to_k.", "to_v.", "to_out.",
             "norm_q.", "norm_k.", "k_norm_und_for_gen.")

# Top-level und modules. "norm." is the und output norm that fed the dropped
# lm_head; nothing reads it, but it stays with the half it was trained in.
_UND_TOPLEVEL = ("embed_tokens.", "norm.")


def is_und_key(key: str) -> bool:
    """True if this checkpoint key belongs to the understanding tower.

    Everything else is gen-side: the add_*_proj attention twins, mlp_moe_gen,
    the *_moe_gen norms, proj_in/proj_out, time_embedder and the audio branch.
    """
    if key.startswith(_UND_TOPLEVEL):
        return True
    if ".self_attn." in key:
        return key.split(".self_attn.", 1)[1].startswith(_UND_ATTN)
    if "moe_gen" in key:
        return False
    # Per-layer und MLP and its two norms; every gen twin carries "moe_gen".
    return (".mlp." in key
            or ".input_layernorm." in key
            or ".post_attention_layernorm." in key)


# ---------------------------------------------------------------------------
# Shard resolution
# ---------------------------------------------------------------------------

# Shard index filenames, in the order they are probed.
_INDEX_FILENAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)
_SINGLE_FILENAMES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
)


def resolve_shard_files(transformer_dir: str):
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
