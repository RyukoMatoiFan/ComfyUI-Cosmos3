"""
cosmos3/avae.py

AVAE (Audio VAE) decoder-only wrapper for Cosmos3 sound generation.

Ports the Cosmos3AVAEAudioTokenizer DECODER from the HuggingFace diffusers
Cosmos3 implementation as a pure-torch module with no diffusers imports.

Module tree (matching the sound_tokenizer checkpoint key names):
  decoder.conv1           — weight_norm Conv1d(64, 5120, 7)
  decoder.block.{0-4}     — Cosmos3AudioDecoderBlock with weight_norm convs
  decoder.snake1          — Snake1d(320)
  decoder.conv2           — weight_norm Conv1d(320, 2, 7, bias=False)

Weight-norm: the checkpoint stores weight_g / weight_v style keys (old style),
so we use torch.nn.utils.weight_norm which produces exactly those parameter names
(conv.weight_g and conv.weight_v — weight_norm registers them as attributes but
they appear under the module's named_parameters() as weight_g and weight_v).

Config (from the model's sound_tokenizer/config.json):
  dec_dim = 320, dec_c_mults = [1,2,4,8,16], dec_strides = [2,4,5,6,8]
  vocoder_input_dim = 64, dec_out_channels = 2
  HOP = prod(dec_strides) = 1920

Strides in Cosmos3AudioDecoder.__init__:
  upsampling_ratios = list(reversed(dec_strides)) = [8,6,5,4,2]
  block[0]: 5120→2560, stride=8
  block[1]: 2560→1280, stride=6
  block[2]: 1280→640,  stride=5
  block[3]: 640→320,   stride=4
  block[4]: 320→320,   stride=2
  (channel_multiples = [1]+dec_c_mults = [1,1,2,4,8,16])

  Per-block channel counts follow:
    input_dim  = channels * channel_multiples[len(strides) - stride_index]
    output_dim = channels * channel_multiples[len(strides) - stride_index - 1]
  output_padding = stride % 2:
  [0, 8%2=0, 6%2=0, 5%2=1, 4%2=0, 2%2=0] → strides [8,6,5,4,2] → [0,0,1,0,0]

Post-processing: clamp(-1, 1) per reference (config dec_final_tanh=false but
reference decode() does .clamp(-1.0, 1.0)).

Lazy wrapper Cosmos3AudioVAE:
  __init__(model_dir) — stores path, no weights loaded
  decode(sound_latent: (B,64,T_s)) -> (B,2,N) float32 cpu in [-1,1]
    loads decoder on demand (strict=True with decoder.* keys),
    runs fp32 on get_torch_device() (fall back to cpu on CUDA error),
    offloads decoder to cpu after each decode call.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)


# ── Building blocks ──────────────────────────────────────────────────────────

class Snake1d(nn.Module):
    """1-D Snake activation with learnable alpha/beta (logscale=True by default)."""

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta  = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        alpha = torch.exp(self.alpha) if self.logscale else self.alpha
        beta  = torch.exp(self.beta)  if self.logscale else self.beta
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)


class Cosmos3AudioResidualUnit(nn.Module):
    """Residual unit: snake → dilated conv → snake → pointwise conv, with skip."""

    def __init__(self, dimension: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1  = weight_norm(
            nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad)
        )
        self.snake2 = Snake1d(dimension)
        self.conv2  = weight_norm(
            nn.Conv1d(dimension, dimension, kernel_size=1)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        out = hidden_state
        out = self.conv1(self.snake1(out))
        out = self.conv2(self.snake2(out))
        padding = (hidden_state.shape[-1] - out.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + out


class Cosmos3AudioDecoderBlock(nn.Module):
    """Decoder block: snake → transposed conv (upsampling) → 3× residual units."""

    def __init__(self, input_dim: int, output_dim: int, stride: int, output_padding: int = 0):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = Cosmos3AudioResidualUnit(output_dim, dilation=1)
        self.res_unit2 = Cosmos3AudioResidualUnit(output_dim, dilation=3)
        self.res_unit3 = Cosmos3AudioResidualUnit(output_dim, dilation=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class Cosmos3AudioDecoder(nn.Module):
    """Full Oobleck-style decoder for Cosmos3 audio.

    Mirrors Cosmos3AudioDecoder in autoencoder_cosmos3_audio.py exactly.
    """

    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list,
        channel_multiples: list,
    ):
        super().__init__()

        # channel_multiples already has the leading 1 prepended by caller:
        # [1] + dec_c_mults → [1, 1, 2, 4, 8, 16]
        strides = upsampling_ratios

        # First conv: input_channels → channels * channel_multiples[-1]
        self.conv1 = weight_norm(
            nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3)
        )

        # Upsampling blocks
        block = []
        for stride_index, stride in enumerate(strides):
            block.append(
                Cosmos3AudioDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            )
        self.block = nn.ModuleList(block)

        output_dim = channels
        self.snake1 = Snake1d(output_dim)
        # No bias on final conv (matches checkpoint — no conv2.bias key)
        self.conv2 = weight_norm(
            nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.block:
            x = layer(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


# ── Lazy wrapper ─────────────────────────────────────────────────────────────

class Cosmos3AudioVAE:
    """Lazy AVAE decoder wrapper.

    Stores only the model directory path on construction.  The decoder weights
    are loaded on the first call to :meth:`decode` and offloaded back to CPU
    after each call.

    Parameters
    ----------
    model_dir : str
        Root directory of the Cosmos3-Nano checkpoint (contains
        ``sound_tokenizer/diffusion_pytorch_model.safetensors``).
    """

    # Decoder config (from sound_tokenizer/config.json)
    _DEC_DIM    = 320
    _DEC_CMULTS = [1, 2, 4, 8, 16]
    _DEC_STRIDES = [2, 4, 5, 6, 8]
    _VOCODER_INPUT_DIM  = 64
    _DEC_OUT_CHANNELS   = 2

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._decoder: Cosmos3AudioDecoder | None = None

    # ------------------------------------------------------------------
    def _build_decoder(self) -> Cosmos3AudioDecoder:
        """Instantiate the decoder (no weights loaded)."""
        channel_multiples = [1] + self._DEC_CMULTS           # [1,1,2,4,8,16]
        upsampling_ratios  = list(reversed(self._DEC_STRIDES))  # [8,6,5,4,2]
        dec = Cosmos3AudioDecoder(
            channels=self._DEC_DIM,
            input_channels=self._VOCODER_INPUT_DIM,
            audio_channels=self._DEC_OUT_CHANNELS,
            upsampling_ratios=upsampling_ratios,
            channel_multiples=channel_multiples,
        )
        return dec

    # ------------------------------------------------------------------
    def decode(self, sound_latent: torch.Tensor) -> torch.Tensor:
        """Decode sound latents to a stereo waveform.

        Parameters
        ----------
        sound_latent : Tensor, shape (B, 64, T_s)
            Raw sound latent from the diffusion model output.

        Returns
        -------
        Tensor, shape (B, 2, N), float32, CPU, clamped to [-1, 1].
        N = T_s * 1920.
        """
        import os
        import comfy.utils
        import comfy.model_management

        ckpt_path = os.path.join(
            self._model_dir, "sound_tokenizer", "diffusion_pytorch_model.safetensors"
        )

        # ── Load weights if not yet loaded ──
        if self._decoder is None:
            logger.info("Cosmos3AudioVAE: loading decoder from %s", ckpt_path)
            sd_full = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

            # Extract only decoder.* keys and strip the "decoder." prefix
            dec_sd = {}
            non_dec_keys = []
            for k, v in sd_full.items():
                if k.startswith("decoder."):
                    dec_sd[k[len("decoder."):]] = v
                else:
                    non_dec_keys.append(k)

            if non_dec_keys:
                logger.info(
                    "Cosmos3AudioVAE: ignoring %d non-decoder keys: %s",
                    len(non_dec_keys), non_dec_keys,
                )

            dec = self._build_decoder()
            missing, unexpected = dec.load_state_dict(dec_sd, strict=True)
            if missing:
                raise RuntimeError(
                    f"Cosmos3AudioVAE: decoder missing keys after strict load: {missing}"
                )
            if unexpected:
                raise RuntimeError(
                    f"Cosmos3AudioVAE: decoder unexpected keys after strict load: {unexpected}"
                )
            logger.info("Cosmos3AudioVAE: decoder loaded successfully (strict, zero missing/unexpected).")
            self._decoder = dec

        # ── Move decoder to compute device ──
        try:
            device = comfy.model_management.get_torch_device()
            self._decoder = self._decoder.float().to(device)
        except Exception as e:
            logger.warning(
                "Cosmos3AudioVAE: failed to move decoder to %s (%s), falling back to CPU.",
                device, e,
            )
            device = torch.device("cpu")
            self._decoder = self._decoder.float().to(device)

        # ── Forward pass ──
        latent_fp32 = sound_latent.float().to(device)
        with torch.no_grad():
            try:
                waveform = self._decoder(latent_fp32)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning(
                        "Cosmos3AudioVAE: CUDA error during decode (%s), retrying on CPU.", e
                    )
                    cpu_device = torch.device("cpu")
                    self._decoder = self._decoder.to(cpu_device)
                    latent_fp32  = latent_fp32.to(cpu_device)
                    waveform     = self._decoder(latent_fp32)
                else:
                    raise

        waveform = waveform.clamp(-1.0, 1.0).float().cpu()

        # ── Offload decoder back to CPU ──
        self._decoder = self._decoder.cpu()

        return waveform
