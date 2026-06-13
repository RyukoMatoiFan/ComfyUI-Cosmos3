"""
Cosmos3OmniTransformer — ComfyUI-native port of the NVIDIA Cosmos3 Omni
transformer (text/und pathway + vision/gen pathway, video + audio).
Pure torch + comfy, no diffusers/transformers imports.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention


# ---------------------------------------------------------------------------
# Sinusoidal timestep projection (no learnable weights — mirrors diffusers Timesteps)
# ---------------------------------------------------------------------------

def _timestep_sinusoidal(t: torch.Tensor, num_channels: int = 256,
                          flip_sin_to_cos: bool = True,
                          downscale_freq_shift: float = 0.0) -> torch.Tensor:
    """Pure-function sinusoidal timestep embedding (equivalent to diffusers Timesteps)."""
    assert num_channels % 2 == 0
    half = num_channels // 2
    # exponents: arange(0, half) / half
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device)
    exponent = exponent / (half - downscale_freq_shift)
    emb = torch.exp(exponent)  # [half]
    emb = t.float()[:, None] * emb[None, :]  # [N, half]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [N, num_channels]
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    return emb


# ---------------------------------------------------------------------------
# mRoPE helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3RotaryEmbedding(nn.Module):
    """
    Cosmos3VLTextRotaryEmbedding — interleaved 3D mRoPE.
    No learnable parameters (inv_freq registered as non-persistent buffer).
    """

    def __init__(self, head_dim: int, rope_theta: float, rope_axes_dim: tuple):
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rope_axes_dim = rope_axes_dim

    def _apply_interleaved_mrope(self, freqs, rope_axes_dim):
        """
        Reorganize chunked [T-block | H-block | W-block] freq layout into interleaved.
        freqs: [3, B, N, head_dim//2]
        returns: [B, N, head_dim//2]
        """
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = rope_axes_dim[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids: torch.Tensor, device, dtype):
        """
        position_ids: [3, B, N] or [3, N] (for a single item)
        Returns (cos, sin) each [B, N, head_dim] or [N, head_dim].
        """
        if position_ids.ndim == 2:
            # [3, N] -> treat B=1 internally
            position_ids = position_ids.unsqueeze(1)  # [3, 1, N]

        inv_freq = self.inv_freq.to(device=device)
        # inv_freq_expanded: [3, B, head_dim//2, 1]
        inv_freq_expanded = (
            inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        # position_ids_expanded: [3, B, 1, N]
        position_ids_expanded = position_ids[:, :, None, :].float()
        # freqs: [3, B, N, head_dim//2]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs, self.rope_axes_dim)  # [B, N, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, N, head_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# ---------------------------------------------------------------------------
# mRoPE position ID builders (ported from pipeline_cosmos3_omni.py)
# ---------------------------------------------------------------------------

def _get_mrope_ids_text(num_tokens: int, temporal_offset):
    """All three axes = monotonically increasing scalar [0..num_tokens-1] + offset."""
    ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()  # [3, num_tokens]
    next_offset = temporal_offset + num_tokens
    return mrope_ids, next_offset


def _get_mrope_ids_vision(grid_t: int, grid_h: int, grid_w: int,
                           temporal_offset,
                           fps: float = 24.0,
                           base_fps: float = 24.0,
                           temporal_compression_factor: int = 4,
                           enable_fps_modulation: bool = True):
    """3D mRoPE position IDs for vision patches."""
    fps_mod = enable_fps_modulation and fps is not None and grid_t > 1
    if fps_mod:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = frame_indices / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.float32)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + float(temporal_offset)
        )

    h_index = torch.arange(grid_h, dtype=torch.float32).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.float32).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)  # [3, N]
    max_pos = mrope_ids.max().item()
    next_offset = math.ceil(max_pos) + 1
    return mrope_ids, next_offset


def _get_mrope_ids_sound(T_s: int,
                          temporal_offset,
                          sound_latent_fps: float = 25.0,
                          base_fps: float = 24.0,
                          enable_fps_modulation: bool = True):
    """
    3D mRoPE position IDs for sound tokens.
    Sound grid = (T_s, 1, 1): temporal_compression_factor_sound = 1.
    Ported exactly from pipeline_cosmos3_omni.py _prepare_sound_segment which calls
    get_3d_mrope_ids_vae_tokens(grid_t=T_s, grid_h=1, grid_w=1, temporal_compression_factor=1).

    fps-modulation formula (enable_fps_modulation=True):
        tps = sound_latent_fps / 1         (temporal_compression_factor_sound = 1)
        base_tps = base_fps / temporal_compression_factor_vision
            BUT the pipeline uses the transformer's base_fps / temporal_compression_factor=1
            for the sound call specifically (see _prepare_sound_segment: tcf=1, no base_tcf override).
        scaled_t = frame_indices / tps * base_tps + temporal_offset
            = frame_indices / sound_latent_fps * base_fps + temporal_offset
    Spatial h/w are reset to 0 (reset_spatial_indices=True by default in pipeline).
    """
    fps_mod = enable_fps_modulation and T_s > 1
    if fps_mod:
        # temporal_compression_factor=1 for sound
        tps = sound_latent_fps / 1
        base_tps = base_fps / 1
        frame_indices = torch.arange(T_s, dtype=torch.float32)
        scaled_t = frame_indices / tps * base_tps + temporal_offset
        t_index = scaled_t  # (T_s,) — h=1, w=1 so no tile needed
    else:
        t_index = torch.arange(T_s, dtype=torch.float32) + float(temporal_offset)

    # h and w both = 0 (grid 1x1 with reset_spatial_indices=True)
    h_index = torch.zeros(T_s, dtype=torch.float32)
    w_index = torch.zeros(T_s, dtype=torch.float32)

    mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)  # [3, T_s]
    max_pos = mrope_ids.max().item()
    next_offset = math.ceil(max_pos) + 1
    return mrope_ids, next_offset


# ---------------------------------------------------------------------------
# MLP (SwiGLU, bias=False)
# ---------------------------------------------------------------------------

class Cosmos3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations
        self.gate_proj = ops.Linear(hidden_size, intermediate_size, bias=False,
                                    device=device, dtype=dtype)
        self.up_proj   = ops.Linear(hidden_size, intermediate_size, bias=False,
                                    device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, hidden_size, bias=False,
                                    device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Dual-pathway packed MoT attention
# ---------------------------------------------------------------------------

class Cosmos3PackedMoTAttention(nn.Module):
    """
    Dual-pathway MoT attention.
    - Understanding (und) path: causal self-attention over text tokens only.
    - Generation (gen) path: full attention over text+gen KVs.
    QK norms applied per-head (head_dim) before RoPE.
    """

    def __init__(self, hidden_size: int, head_dim: int,
                 num_attention_heads: int, num_key_value_heads: int,
                 rms_norm_eps: float,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        # Understanding pathway
        self.to_q   = ops.Linear(hidden_size, num_attention_heads * head_dim, bias=False,
                                  device=device, dtype=dtype)
        self.to_k   = ops.Linear(hidden_size, num_key_value_heads * head_dim, bias=False,
                                  device=device, dtype=dtype)
        self.to_v   = ops.Linear(hidden_size, num_key_value_heads * head_dim, bias=False,
                                  device=device, dtype=dtype)
        self.to_out = ops.Linear(num_attention_heads * head_dim, hidden_size, bias=False,
                                  device=device, dtype=dtype)
        self.norm_q = ops.RMSNorm(head_dim, eps=rms_norm_eps,
                                   elementwise_affine=True, device=device, dtype=dtype)
        self.norm_k = ops.RMSNorm(head_dim, eps=rms_norm_eps,
                                   elementwise_affine=True, device=device, dtype=dtype)

        # Generation pathway
        self.add_q_proj  = ops.Linear(hidden_size, num_attention_heads * head_dim, bias=False,
                                       device=device, dtype=dtype)
        self.add_k_proj  = ops.Linear(hidden_size, num_key_value_heads * head_dim, bias=False,
                                       device=device, dtype=dtype)
        self.add_v_proj  = ops.Linear(hidden_size, num_key_value_heads * head_dim, bias=False,
                                       device=device, dtype=dtype)
        self.to_add_out  = ops.Linear(num_attention_heads * head_dim, hidden_size, bias=False,
                                       device=device, dtype=dtype)
        self.norm_added_q = ops.RMSNorm(head_dim, eps=rms_norm_eps,
                                         elementwise_affine=True, device=device, dtype=dtype)
        self.norm_added_k = ops.RMSNorm(head_dim, eps=rms_norm_eps,
                                         elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, und_seq, gen_seq,
                cos_und, sin_und, cos_gen, sin_gen):
        """
        und_seq: [L_text, D]
        gen_seq: [L_gen, D]
        cos/sin_und: [L_text, head_dim]
        cos/sin_gen: [L_gen, head_dim]
        Returns (und_out, gen_out) each same shape as inputs.
        """
        H_q  = self.num_attention_heads
        H_kv = self.num_key_value_heads
        d    = self.head_dim
        n_rep = H_q // H_kv

        # --- Projections ---
        q_und = self.to_q(und_seq).view(-1, H_q, d)   # [L_t, H_q, d]
        k_und = self.to_k(und_seq).view(-1, H_kv, d)  # [L_t, H_kv, d]
        v_und = self.to_v(und_seq).view(-1, H_kv, d)

        q_gen = self.add_q_proj(gen_seq).view(-1, H_q, d)
        k_gen = self.add_k_proj(gen_seq).view(-1, H_kv, d)
        v_gen = self.add_v_proj(gen_seq).view(-1, H_kv, d)

        # --- QK norm (per-head, on head_dim) ---
        q_und = self.norm_q(q_und)
        k_und = self.norm_k(k_und)
        q_gen = self.norm_added_q(q_gen)
        k_gen = self.norm_added_k(k_gen)

        # --- RoPE application ---
        cos_und_ = cos_und.unsqueeze(1)  # [L_t, 1, head_dim]
        sin_und_ = sin_und.unsqueeze(1)
        q_und = q_und * cos_und_ + _rotate_half(q_und) * sin_und_
        k_und = k_und * cos_und_ + _rotate_half(k_und) * sin_und_

        cos_gen_ = cos_gen.unsqueeze(1)  # [L_g, 1, head_dim]
        sin_gen_ = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen_ + _rotate_half(q_gen) * sin_gen_
        k_gen = k_gen * cos_gen_ + _rotate_half(k_gen) * sin_gen_

        # --- Causal text attention (und pathway, GQA via repeat) ---
        # optimized_attention expects [B, L, H*d] flat
        L_t = und_seq.shape[0]
        # repeat KV heads for GQA
        k_und_rep = k_und.repeat_interleave(n_rep, dim=1)  # [L_t, H_q, d]
        v_und_rep = v_und.repeat_interleave(n_rep, dim=1)

        q_und_flat = q_und.view(1, L_t, H_q * d)
        k_und_flat = k_und_rep.view(1, L_t, H_q * d)
        v_und_flat = v_und_rep.view(1, L_t, H_q * d)

        # Causal mask for text
        causal_mask = torch.triu(
            torch.full((L_t, L_t), float('-inf'), dtype=q_und.dtype, device=q_und.device),
            diagonal=1
        )
        # Use pytorch SDPA for causal text attention (mask-based for compatibility)
        q_und_4d = q_und.unsqueeze(0).permute(0, 2, 1, 3)  # [1, H_q, L_t, d]
        k_und_4d = k_und_rep.unsqueeze(0).permute(0, 2, 1, 3)
        v_und_4d = v_und_rep.unsqueeze(0).permute(0, 2, 1, 3)
        causal_out_4d = F.scaled_dot_product_attention(
            q_und_4d, k_und_4d, v_und_4d, is_causal=True
        )
        causal_out = causal_out_4d.permute(0, 2, 1, 3).reshape(1, L_t, H_q * d)

        # --- Full attention (gen pathway, KV = concat(und, gen)) ---
        L_g = gen_seq.shape[0]
        L_all = L_t + L_g
        all_k = torch.cat([k_und, k_gen], dim=0)  # [L_all, H_kv, d]
        all_v = torch.cat([v_und, v_gen], dim=0)

        # Repeat KV heads
        all_k_rep = all_k.repeat_interleave(n_rep, dim=1)  # [L_all, H_q, d]
        all_v_rep = all_v.repeat_interleave(n_rep, dim=1)

        q_gen_flat  = q_gen.view(1, L_g, H_q * d)
        k_all_flat  = all_k_rep.view(1, L_all, H_q * d)
        v_all_flat  = all_v_rep.view(1, L_all, H_q * d)

        full_out = optimized_attention(q_gen_flat, k_all_flat, v_all_flat, heads=H_q)
        # full_out: [1, L_g, H_q*d]

        und_out = self.to_out(causal_out.squeeze(0))    # [L_t, D]
        gen_out = self.to_add_out(full_out.squeeze(0))  # [L_g, D]
        return und_out, gen_out


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Cosmos3DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int,
                 num_attention_heads: int, num_key_value_heads: int,
                 intermediate_size: int, rms_norm_eps: float,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations
        self.self_attn = Cosmos3PackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            device=device, dtype=dtype, operations=ops,
        )
        # Understanding MLP
        self.mlp = Cosmos3MLP(hidden_size, intermediate_size,
                               device=device, dtype=dtype, operations=ops)
        # Generation MLP
        self.mlp_moe_gen = Cosmos3MLP(hidden_size, intermediate_size,
                                       device=device, dtype=dtype, operations=ops)

        self.input_layernorm = ops.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True,
            device=device, dtype=dtype)
        self.input_layernorm_moe_gen = ops.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True,
            device=device, dtype=dtype)
        self.post_attention_layernorm = ops.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True,
            device=device, dtype=dtype)
        self.post_attention_layernorm_moe_gen = ops.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True,
            device=device, dtype=dtype)

    def forward(self, und_seq, gen_seq, cos_und, sin_und, cos_gen, sin_gen):
        # Pre-norm
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        # Dual-pathway attention
        und_attn, gen_attn = self.self_attn(
            und_norm, gen_norm, cos_und, sin_und, cos_gen, sin_gen
        )

        # Residual
        und_seq = und_seq + und_attn
        gen_seq = gen_seq + gen_attn

        # MLP + residual
        und_seq = und_seq + self.mlp(self.post_attention_layernorm(und_seq))
        gen_seq = gen_seq + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(gen_seq))

        return und_seq, gen_seq


# ---------------------------------------------------------------------------
# TimestepEmbedding (equivalent to diffusers TimestepEmbedding)
# ---------------------------------------------------------------------------

class Cosmos3TimestepEmbedding(nn.Module):
    """
    Two-layer MLP: Linear(in_channels, out_channels) + SiLU + Linear(out_channels, out_channels).
    Key names: linear_1, linear_2 (matching checkpoint).
    Kept in fp32 per reference (_keep_in_fp32_modules).
    """

    def __init__(self, in_channels: int, time_embed_dim: int,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations
        # Always build in fp32 — these are kept in fp32
        self.linear_1 = ops.Linear(in_channels, time_embed_dim, bias=True,
                                    device=device, dtype=dtype)
        self.linear_2 = ops.Linear(time_embed_dim, time_embed_dim, bias=True,
                                    device=device, dtype=dtype)

    def forward(self, x):
        # Run in the linear's compute dtype. fp8 storage has no matmul kernel, so
        # fall back to bf16 (manual_cast upcasts the fp8 weight to match).
        weight_dtype = self.linear_1.weight.dtype if self.linear_1.weight is not None else x.dtype
        if weight_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            weight_dtype = torch.bfloat16
        x = x.to(weight_dtype)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x.float()


# ---------------------------------------------------------------------------
# Main transformer
# ---------------------------------------------------------------------------

class Cosmos3OmniTransformer(nn.Module):
    """
    ComfyUI port of the diffusers Cosmos3OmniTransformer (video + audio).
    """

    def __init__(
        self,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
        # Config defaults from transformer/config.json
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 12288,
        latent_channel: int = 48,
        latent_patch_size: int = 2,
        vocab_size: int = 151936,
        rope_theta: float = 5000000.0,
        mrope_section: tuple = (24, 20, 20),
        base_fps: int = 24,
        enable_fps_modulation: bool = True,
        timestep_scale: float = 0.001,
        mrope_margin: int = 15000,
        rms_norm_eps: float = 1e-6,
        # accept but ignore rope_scaling dict for caller convenience
        rope_scaling=None,
        patch_latent_dim: int = 192,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype

        # Store config
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.latent_channel = latent_channel
        self.latent_patch_size = latent_patch_size
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta
        self.base_fps = base_fps
        self.enable_fps_modulation = enable_fps_modulation
        self.timestep_scale = timestep_scale
        self.mrope_margin = mrope_margin
        self.rms_norm_eps = rms_norm_eps
        self.patch_latent_dim = patch_latent_dim  # p^2 * latent_channel

        # Extract mrope_section from rope_scaling dict if provided
        if rope_scaling is not None and "mrope_section" in rope_scaling:
            mrope_section = tuple(rope_scaling["mrope_section"])
        self.mrope_section = mrope_section

        ops = operations

        # Text embedding
        self.embed_tokens = ops.Embedding(vocab_size, hidden_size,
                                           device=device, dtype=dtype)

        # Transformer layers
        self.layers = nn.ModuleList([
            Cosmos3DecoderLayer(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                device=device, dtype=dtype, operations=ops,
            )
            for _ in range(num_hidden_layers)
        ])

        # Final norms
        self.norm         = ops.RMSNorm(hidden_size, eps=rms_norm_eps,
                                         elementwise_affine=True, device=device, dtype=dtype)
        self.norm_moe_gen = ops.RMSNorm(hidden_size, eps=rms_norm_eps,
                                         elementwise_affine=True, device=device, dtype=dtype)

        # Rotary embedding (no learnable weights)
        self.rotary_emb = Cosmos3RotaryEmbedding(
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_axes_dim=mrope_section,
        )

        # Vision in/out projection
        self.proj_in  = ops.Linear(patch_latent_dim, hidden_size, bias=True,
                                    device=device, dtype=dtype)
        self.proj_out = ops.Linear(hidden_size, patch_latent_dim, bias=True,
                                    device=device, dtype=dtype)

        # Timestep embedding (time_proj is weight-free sinusoidal; time_embedder has weights)
        # time_proj has NO weight tensors in checkpoint — sinusoidal function only
        # time_embedder.linear_1 / linear_2 match checkpoint keys
        self.time_embedder = Cosmos3TimestepEmbedding(
            in_channels=256,
            time_embed_dim=hidden_size,
            device=device, dtype=dtype, operations=ops,
        )

        # Audio generation modules — exact checkpoint key names from keys_transformer.txt:
        #   audio_proj_in.weight [4096,64], audio_proj_in.bias [4096]
        #   audio_proj_out.weight [64,4096], audio_proj_out.bias [64]
        #   audio_modality_embed [4096]
        self.audio_proj_in  = ops.Linear(64, hidden_size, bias=True,
                                          device=device, dtype=dtype)
        self.audio_proj_out = ops.Linear(hidden_size, 64, bias=True,
                                          device=device, dtype=dtype)
        # nn.Parameter so the key name is "audio_modality_embed" (not .weight)
        self.audio_modality_embed = nn.Parameter(
            torch.zeros(hidden_size, device=device,
                        dtype=dtype if dtype is not None else torch.float32)
        )

    # ------------------------------------------------------------------
    # Patchify / unpatchify (ported exactly from reference)
    # ------------------------------------------------------------------

    def _patchify(self, latent: torch.Tensor):
        """
        latent: [C, T, H, W]
        Returns packed: [T * Hp * Wp, p^2 * C], padded_hw: (H_padded, W_padded), (Hp, Wp)
        """
        p = self.latent_patch_size
        C = self.latent_channel
        _, t, h_actual, w_actual = latent.shape

        h_padded = ((h_actual + p - 1) // p) * p
        w_padded = ((w_actual + p - 1) // p) * p
        if h_padded != h_actual or w_padded != w_actual:
            padded = torch.zeros(C, t, h_padded, w_padded,
                                  device=latent.device, dtype=latent.dtype)
            padded[:, :, :h_actual, :w_actual] = latent
            latent = padded

        Hp = h_padded // p
        Wp = w_padded // p
        latent = latent.reshape(C, t, Hp, p, Wp, p)
        # einsum "cthpwq -> thwpqc" then flatten spatial dims
        latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * C)
        return latent, (h_actual, w_actual), (Hp, Wp)

    def _unpatchify(self, patches: torch.Tensor,
                     t: int, Hp: int, Wp: int,
                     h_orig: int, w_orig: int) -> torch.Tensor:
        """
        patches: [N, p^2 * C]  (N = t * Hp * Wp)
        Returns [C, t, h_orig, w_orig]
        """
        p = self.latent_patch_size
        C = self.latent_channel
        patches = patches.reshape(t, Hp, Wp, p, p, C)
        latent = torch.einsum("thwpqc->cthpwq", patches)
        latent = latent.reshape(C, t, Hp * p, Wp * p)
        return latent[:, :, :h_orig, :w_orig]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                context: torch.Tensor,
                cosmos3_fps=None,
                cosmos3_cond_latent=None,
                cosmos3_cond_mask=None,
                cosmos3_sound_latent=None,
                transformer_options: dict = {},
                **kwargs):
        """
        x:                    (B, 48, T, H, W) noisy latent
        timesteps:            (B,) comfy sigma s in (0, 1]
        context:              (B, n_tokens, 1) float token IDs
        cosmos3_fps:          optional float fps (default 24)
        cosmos3_cond_latent:  (B, 48, T, H, W) or (1, 48, T, H, W) clean latent for I2V
        cosmos3_cond_mask:    (B, 1, T, 1, 1) or (1, 1, T, 1, 1) mask: 1 = conditioned frame
        cosmos3_sound_latent: optional (B, 64, T_s) RAW sound latent
        Returns:
          - Without sound: velocity (B, 48, T, H, W)   [backward-compatible]
          - With sound:    tuple (v_vision (B,48,T,H,W), v_sound (B,64,T_s))
        """
        B, C, T, H, W = x.shape
        fps = float(cosmos3_fps) if cosmos3_fps is not None else float(self.base_fps)
        # Compute dtype = incoming activation dtype (bf16 under fp8 manual_cast), NOT
        # self.dtype which is the fp8 storage dtype.
        compute_dtype = x.dtype

        has_sound = cosmos3_sound_latent is not None

        # Parse token IDs: (B, n_tokens, 1) float -> list of 1D long tensors
        token_ids_list = []
        for b in range(B):
            ids = context[b, :, 0].round().long()  # [n_tokens]
            token_ids_list.append(ids)

        # comfy sigma s is the flow sigma fed to the timestep embedding.
        sigmas = timesteps  # [B]

        # Output tensors
        out = torch.zeros_like(x)
        if has_sound:
            T_s = cosmos3_sound_latent.shape[2]
            out_sound = torch.zeros_like(cosmos3_sound_latent)

        for b in range(B):
            s_b = sigmas[b].clamp(min=1e-4)  # comfy flow sigma s
            # timestep embedding input = s (== s*1000 * timestep_scale, the reference form)
            t_emb_input = (s_b * 1000.0) * self.timestep_scale

            x_b = x[b]   # [C, T, H, W]

            # Conditioned-frame handling (I2V, video only)
            cond_mask_b = None
            cond_latent_b = None
            if cosmos3_cond_mask is not None and cosmos3_cond_latent is not None:
                # mask: (B,1,T,1,1) or (1,1,T,1,1)
                mask_b = cosmos3_cond_mask[min(b, cosmos3_cond_mask.shape[0]-1), 0, :, 0, 0]  # [T]
                cond_lat_b = cosmos3_cond_latent[min(b, cosmos3_cond_latent.shape[0]-1)]  # [C,T,H,W]
                cond_frames = (mask_b > 0.5).nonzero(as_tuple=True)[0]  # indices of cond frames
                if len(cond_frames) > 0:
                    cond_mask_b = cond_frames
                    cond_latent_b = cond_lat_b
                    # Substitute cond values into x_b for those frames
                    x_b = x_b.clone()
                    x_b[:, cond_frames] = cond_lat_b[:, cond_frames]

            # Patchify vision
            patches_b, (h_orig, w_orig), (Hp, Wp) = self._patchify(x_b)
            # patches_b: [T*Hp*Wp, p^2*C]

            # Determine noisy vs conditioned frame patch indices
            all_frames = torch.arange(T, device=x.device)
            if cond_mask_b is not None:
                cond_frame_set = set(cond_mask_b.tolist())
                noisy_frames = torch.tensor(
                    [f for f in range(T) if f not in cond_frame_set],
                    dtype=torch.long, device=x.device
                )
            else:
                noisy_frames = all_frames
                cond_frame_set = set()

            # Project all vision patches
            patches_proj = self.proj_in(patches_b.to(compute_dtype))  # [T*Hp*Wp, D]

            # Compute timestep embedding (in fp32, result cast to compute_dtype)
            t_emb_vec = _timestep_sinusoidal(
                t_emb_input.unsqueeze(0), num_channels=256,
                flip_sin_to_cos=True, downscale_freq_shift=0.0
            )  # [1, 256] float32
            time_embed = self.time_embedder(t_emb_vec).squeeze(0)  # [D] float32
            time_embed = time_embed.to(compute_dtype)

            # Add timestep embedding to noisy-frame vision patches only
            spatial_numel = Hp * Wp
            for fi in noisy_frames.tolist():
                start = fi * spatial_numel
                end   = start + spatial_numel
                patches_proj[start:end] = patches_proj[start:end] + time_embed

            # Build joint sequence
            token_ids = token_ids_list[b]   # [L_text]
            L_text = token_ids.shape[0]

            text_hidden = self.embed_tokens(token_ids.to(x.device)).to(compute_dtype)  # [L_text, D]
            L_vision = patches_proj.shape[0]  # T*Hp*Wp

            und_len = L_text

            # Build mRoPE position IDs
            # Text: offset starts at 0
            text_mrope_ids, next_offset = _get_mrope_ids_text(L_text, temporal_offset=0)
            # Vision AND sound both use the same temporal offset base = und_len + mrope_margin
            # (they are temporal siblings, not sequential — see _prepare_text_segment in pipeline)
            modality_temporal_offset = L_text + self.mrope_margin
            vision_mrope_ids, _ = _get_mrope_ids_vision(
                grid_t=T,
                grid_h=Hp,
                grid_w=Wp,
                temporal_offset=modality_temporal_offset,
                fps=fps,
                base_fps=float(self.base_fps),
                temporal_compression_factor=4,
                enable_fps_modulation=self.enable_fps_modulation,
            )

            if has_sound:
                # Sound tokens: all noisy
                sound_b = cosmos3_sound_latent[b]  # [64, T_s]
                T_s_b = sound_b.shape[1]

                # audio_proj_in: (T_s, 64) -> (T_s, D), then + audio_modality_embed
                sound_tokens = self.audio_proj_in(
                    sound_b.T.to(compute_dtype)
                )  # [T_s, D]
                audio_mod_emb = self.audio_modality_embed.to(dtype=compute_dtype, device=sound_tokens.device)
                sound_tokens = sound_tokens + audio_mod_emb  # [T_s, D]

                # Add timestep embedding to ALL sound tokens (all-noisy)
                # sound_timesteps uses the same sigma_raw as vision
                # all T_s tokens get the same time_embed (noisy_frame_indexes = [0..T_s-1])
                sound_tokens = sound_tokens + time_embed.unsqueeze(0)  # [T_s, D]

                # Sound mRoPE: same temporal offset base as vision
                # temporal_compression_factor_sound = 1, sound_latent_fps = 25
                sound_mrope_ids, _ = _get_mrope_ids_sound(
                    T_s=T_s_b,
                    temporal_offset=modality_temporal_offset,
                    sound_latent_fps=25.0,
                    base_fps=float(self.base_fps),
                    enable_fps_modulation=self.enable_fps_modulation,
                )

                # gen sequence = [vision | sound]
                gen_hidden = torch.cat([patches_proj, sound_tokens], dim=0)  # [L_vision+T_s, D]
                vision_sound_mrope_ids = torch.cat(
                    [vision_mrope_ids.to(x.device), sound_mrope_ids.to(x.device)], dim=1
                )  # [3, L_vision + T_s]

                # Full position_ids: [3, L_text + L_vision + T_s]
                position_ids = torch.cat(
                    [text_mrope_ids.to(x.device), vision_sound_mrope_ids], dim=1
                )
            else:
                gen_hidden = patches_proj  # [L_vision, D]
                position_ids = torch.cat(
                    [text_mrope_ids.to(x.device),
                     vision_mrope_ids.to(x.device)], dim=1
                )  # [3, L_text + L_vision]

            L_gen = gen_hidden.shape[0]

            # Compute rotary embeddings for full sequence
            cos, sin = self.rotary_emb(
                position_ids=position_ids.unsqueeze(1) if position_ids.ndim == 2 else position_ids,
                device=x.device,
                dtype=compute_dtype,
            )
            # cos, sin: [1, total_len, head_dim]
            cos = cos.squeeze(0)  # [total_len, head_dim]
            sin = sin.squeeze(0)

            cos_und = cos[:L_text]
            sin_und = sin[:L_text]
            cos_gen = cos[L_text:]
            sin_gen = sin[L_text:]

            # Run transformer layers
            und_seq = text_hidden  # [L_text, D]
            gen_seq = gen_hidden   # [L_gen, D]  (vision + optional sound)

            for layer in self.layers:
                und_seq, gen_seq = layer(und_seq, gen_seq,
                                          cos_und, sin_und, cos_gen, sin_gen)

            # Final norms (und_seq discarded — text output not needed for video)
            gen_out = self.norm_moe_gen(gen_seq)  # [L_gen, D]

            # --- Vision output ---
            vision_gen_out = gen_out[:L_vision]  # [L_vision, D]
            pred_patches = self.proj_out(vision_gen_out)  # [L_vision, p^2*C]
            pred_vol = self._unpatchify(pred_patches, T, Hp, Wp, h_orig, w_orig)
            # pred_vol: [C, T, h_orig, w_orig]
            out[b] = pred_vol

            # For conditioned frames, override velocity so denoised = cond_latent
            if cond_mask_b is not None and len(cond_mask_b) > 0:
                # velocity formula: v = (x_orig - cond_latent) / s
                # so that denoised = x - s*v = x - (x_orig - cond_latent) = cond_latent
                x_orig_b = x[b]  # original noisy (before substitution)
                cond_lat_b = cond_latent_b
                for fi in cond_mask_b.tolist():
                    out[b, :, fi] = (x_orig_b[:, fi] - cond_lat_b[:, fi]) / s_b.clamp(min=1e-4)

            # --- Sound output ---
            if has_sound:
                sound_gen_out = gen_out[L_vision:]  # [T_s, D]
                # audio_proj_out: (T_s, D) -> (T_s, 64), then transpose to (64, T_s)
                pred_sound_b = self.audio_proj_out(sound_gen_out).T  # [64, T_s]
                out_sound[b] = pred_sound_b

        if has_sound:
            return out, out_sound
        return out
