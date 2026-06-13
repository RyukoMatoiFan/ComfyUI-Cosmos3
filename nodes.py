"""
ComfyUI-Cosmos3 — node layer (V1 API)
All six Cosmos3 nodes live here; __init__.py re-exports the mappings.
"""

import os
import torch
import folder_paths
import node_helpers
import comfy.model_management
import comfy.utils

# ---------------------------------------------------------------------------
# Register the cosmos3 model folder at import time so ComfyUI can find models
# placed under <comfyui>/models/cosmos3/ as well as absolute paths.
# ---------------------------------------------------------------------------
_cosmos3_models_dir = os.path.join(folder_paths.models_dir, "cosmos3")
folder_paths.add_model_folder_path("cosmos3", _cosmos3_models_dir)


# Lazy imports from the cosmos3 package — resolved at call time so a missing
# dependency does not break ComfyUI startup.

def _get_loader():
    from .cosmos3.loader import load_cosmos3_model, load_cosmos3_vae, load_cosmos3_tokenizer
    return load_cosmos3_model, load_cosmos3_vae, load_cosmos3_tokenizer


def _get_compute_sigmas():
    from .cosmos3.scheduler import compute_cosmos3_sigmas
    return compute_cosmos3_sigmas


def _get_avae():
    from .cosmos3.avae import Cosmos3AudioVAE
    return Cosmos3AudioVAE


def _get_av_pack():
    from .cosmos3.av_pack import sound_token_count, extra_frames, audio_sample_count, unpack_sound
    return sound_token_count, extra_frames, audio_sample_count, unpack_sound


# ---------------------------------------------------------------------------
# 1. Cosmos3Loader
# ---------------------------------------------------------------------------

class Cosmos3Loader:
    """
    Loads the Cosmos3-Nano transformer, VAE and text tokenizer from a local
    model directory.  The directory can be an absolute path or a bare name
    that will be resolved inside the registered cosmos3 model folders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {
                    "default": "Cosmos3-Nano",
                    "tooltip": "Path to the Cosmos3-Nano model directory "
                               "(contains transformer/, vae/, text_tokenizer/ sub-folders). "
                               "Accepts absolute path or a bare directory name resolvable "
                               "inside models/cosmos3/.",
                }),
                "weight_dtype": (["default", "fp8_e4m3fn"], {
                    "tooltip": "default = bfloat16 (native). "
                               "fp8_e4m3fn casts linear weights to fp8 to save VRAM.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "COSMOS3_TEXT_ENCODER", "VAE", "COSMOS3_AUDIO_VAE")
    RETURN_NAMES = ("model", "text_encoder", "vae", "audio_vae")
    FUNCTION = "load"
    CATEGORY = "Cosmos3"

    def load(self, model_dir: str, weight_dtype: str):
        # Resolve bare directory name against registered cosmos3 folders
        if not os.path.isabs(model_dir) or not os.path.isdir(model_dir):
            cosmos3_paths = folder_paths.get_folder_paths("cosmos3")
            for base in cosmos3_paths:
                candidate = os.path.join(base, model_dir)
                if os.path.isdir(candidate):
                    model_dir = candidate
                    break

        load_model, load_vae, load_tokenizer = _get_loader()
        model = load_model(model_dir, weight_dtype)
        vae = load_vae(model_dir)
        text_encoder = load_tokenizer(model_dir)
        Cosmos3AudioVAE = _get_avae()
        audio_vae = Cosmos3AudioVAE(model_dir)
        return (model, text_encoder, vae, audio_vae)


# ---------------------------------------------------------------------------
# 2. Cosmos3TextEncode
# ---------------------------------------------------------------------------

class Cosmos3TextEncode:
    """
    Tokenises a text prompt through the Cosmos3 Qwen2 chat-template pipeline
    and returns a CONDITIONING tensor compatible with ComfyUI's CFGGuider.

    The prompt accepts plain text or a Cosmos3 JSON-upsampled prompt string
    (the JSON is forwarded verbatim to the tokeniser which handles it as text).

    Resolution and duration metadata sentences are optionally appended before
    tokenisation (add_resolution_template / add_duration_template).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("COSMOS3_TEXT_ENCODER",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Accepts plain text or a Cosmos3 JSON-upsampled prompt.",
                }),
                "width": ("INT", {
                    "default": 1280, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video width in pixels (must be multiple of 16).",
                }),
                "height": ("INT", {
                    "default": 720, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video height in pixels (must be multiple of 16).",
                }),
                "num_frames": ("INT", {
                    "default": 189, "min": 1, "max": 401, "step": 4,
                    "tooltip": "Number of video frames. 189 = 7.9 s @ 24 fps. "
                               "Use 93 frames (~3.9 s) for lower VRAM.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Frames per second used in the duration metadata template.",
                }),
                "add_resolution_template": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append 'This video is of {H}x{W} resolution.' to the prompt.",
                }),
                "add_duration_template": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append duration and FPS sentence to the prompt.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Cosmos3"

    def encode(self, text_encoder, prompt, width, height, num_frames, fps,
               add_resolution_template, add_duration_template):
        token_ids = text_encoder.tokenize(
            prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            add_resolution_template=add_resolution_template,
            add_duration_template=add_duration_template,
        )
        # ids_tensor: float32 (1, n_tokens, 1) — stores integer IDs as floats
        ids_tensor = torch.tensor(token_ids, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        extras = {"cosmos3_fps": float(fps)}
        conditioning = [[ids_tensor, extras]]
        return (conditioning,)


# ---------------------------------------------------------------------------
# 3. Cosmos3DefaultNegativePrompt
# ---------------------------------------------------------------------------

class Cosmos3DefaultNegativePrompt:
    """
    Reads the bundled negative_prompt.json from the model's assets/ directory
    and returns it as a STRING that can be wired into Cosmos3TextEncode's
    prompt input.

    If the file is absent (e.g. during testing), returns an empty string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("COSMOS3_TEXT_ENCODER",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative_prompt",)
    FUNCTION = "get_negative_prompt"
    CATEGORY = "Cosmos3"

    def get_negative_prompt(self, text_encoder):
        neg = text_encoder.default_negative_prompt()
        return (neg,)


# ---------------------------------------------------------------------------
# 4. Cosmos3EmptyLatentVideo
# ---------------------------------------------------------------------------

class Cosmos3EmptyLatentVideo:
    """
    Creates an empty (zero-filled) latent tensor with the correct shape for
    Cosmos3-Nano: [B, 48, T_lat, H//16, W//16] where
    T_lat = (length - 1) // 4 + 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1280, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video width (must be multiple of 16).",
                }),
                "height": ("INT", {
                    "default": 720, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video height (must be multiple of 16).",
                }),
                "length": ("INT", {
                    "default": 93, "min": 1, "max": 401, "step": 4,
                    "tooltip": "Number of video frames. Latent temporal dim = (length-1)//4+1.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 4096,
                    "tooltip": "Batch size.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Cosmos3"

    def generate(self, width, height, length, batch_size):
        latent_t = (length - 1) // 4 + 1
        latent_h = height // 16
        latent_w = width // 16
        samples = torch.zeros(
            [batch_size, 48, latent_t, latent_h, latent_w],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": samples},)


# ---------------------------------------------------------------------------
# 5. Cosmos3ImageToVideo
# ---------------------------------------------------------------------------

class Cosmos3ImageToVideo:
    """
    First-frame image conditioning node for Cosmos3 image-to-video generation.

    Mirrors WanImageToVideo: upscales the input image to (width, height), then
    fills all frames with the image (repeat-fill).
    VAE-encodes to get cond_latent, marks only frame 0 as conditioned (mask=1),
    and attaches both latent and mask to positive and negative conditioning.

    Returns the modified conditioning tensors and an empty latent for the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 1280, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video width (must be multiple of 16).",
                }),
                "height": ("INT", {
                    "default": 720, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video height (must be multiple of 16).",
                }),
                "length": ("INT", {
                    "default": 93, "min": 1, "max": 401, "step": 4,
                    "tooltip": "Number of video frames.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 4096,
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "Cosmos3"

    def encode(self, positive, negative, vae, image, width, height, length, batch_size):
        # --- upscale the first input image to (width, height) ---
        # image is (B_img, H_src, W_src, C) in [0,1]; pick first frame
        img = comfy.utils.common_upscale(
            image[:1].movedim(-1, 1),   # (1, C, H, W)
            width, height,
            "bilinear", "center",
        ).movedim(1, -1)                # (1, H, W, C)

        # --- fill all `length` frames with the conditioning image ---
        # pixel_video: (length, H, W, C) in [0,1]
        pixel_video = img.expand(length, -1, -1, -1).clone()

        # --- VAE encode ---
        # vae.encode expects (T, H, W, C) pixel tensor with values in [0,1]
        # Slice to 3 channels for safety (mirrors WanImageToVideo)
        cond_latent = vae.encode(pixel_video[:, :, :, :3])
        # cond_latent: (1, 48, T_lat, latent_h, latent_w)

        # --- conditioning mask: shape (1, 1, T_lat, 1, 1), frame 0 = 1.0 ---
        latent_t = cond_latent.shape[2]
        mask = torch.zeros(
            (1, 1, latent_t, 1, 1),
            dtype=cond_latent.dtype,
            device=cond_latent.device,
        )
        mask[:, :, 0] = 1.0   # mark first latent frame as clean / conditioned

        # --- attach to conditioning ---
        cond_values = {
            "cosmos3_cond_latent": cond_latent,
            "cosmos3_cond_mask": mask,
        }
        positive = node_helpers.conditioning_set_values(positive, cond_values)
        negative = node_helpers.conditioning_set_values(negative, cond_values)

        # --- empty latent for the denoising sampler ---
        latent_h = height // 16
        latent_w = width // 16
        empty_samples = torch.zeros(
            [batch_size, 48, latent_t, latent_h, latent_w],
            device=comfy.model_management.intermediate_device(),
        )

        return (positive, negative, {"samples": empty_samples})


# ---------------------------------------------------------------------------
# 6. Cosmos3Scheduler
# ---------------------------------------------------------------------------

class Cosmos3Scheduler:
    """
    Cosmos3 UniPC flow schedule: uniform sigmas over [1, 0] with a discrete shift
    `s' = shift*s / (1 + (shift-1)*s)`. Output feeds SamplerCustomAdvanced.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 35, "min": 1, "max": 200,
                    "tooltip": "Number of denoising steps.",
                }),
                "flow_shift": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Discrete flow shift, resolution-dependent: "
                               "10 @ 720p, 5 @ 480p, 3 @ 256p. 1.0 = no shift.",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoising strength (1.0 = full denoise).",
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "Cosmos3"

    def get_sigmas(self, steps: int, flow_shift: float, denoise: float):
        compute_sigmas = _get_compute_sigmas()
        sigmas = compute_sigmas(steps=steps, flow_shift=flow_shift, denoise=denoise)
        return (sigmas,)


# ---------------------------------------------------------------------------
# 7. Cosmos3EmptyAVLatentVideo
# ---------------------------------------------------------------------------

class Cosmos3EmptyAVLatentVideo:
    """
    Creates a packed audio-video latent for joint AV denoising.

    Sound latent (B, 64, T_sound) is packed into extra latent frames appended
    along the T axis of the video latent (B, 48, T_video_lat, H//16, W//16).
    The resulting packed tensor can be passed directly to SamplerCustomAdvanced.

    fps and length MUST match the values used in Cosmos3TextEncode and CreateVideo.

    For I2V + audio: chain Cosmos3ImageToVideo FIRST (its positive/negative conds
    carry the I2V conditioning), then pass those conds into this node's positive/
    negative inputs.  Use THIS node's latent output (packed AV) for the sampler;
    DISCARD the latent produced by Cosmos3ImageToVideo.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning from Cosmos3TextEncode "
                               "(or Cosmos3ImageToVideo for I2V+audio).",
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning from Cosmos3TextEncode "
                               "(or Cosmos3ImageToVideo for I2V+audio).",
                }),
                "width": ("INT", {
                    "default": 1280, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video width (must be multiple of 16).",
                }),
                "height": ("INT", {
                    "default": 720, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Output video height (must be multiple of 16).",
                }),
                "length": ("INT", {
                    "default": 93, "min": 5, "max": 401, "step": 4,
                    "tooltip": "Number of video frames. "
                               "Must match num_frames in Cosmos3TextEncode and fps in CreateVideo.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Frames per second. "
                               "Must match fps in Cosmos3TextEncode and CreateVideo.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 4096,
                    "tooltip": "Batch size.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "Cosmos3"

    def generate(self, positive, negative, width, height, length, fps, batch_size):
        sound_token_count, extra_frames_fn, audio_sample_count, _unpack = _get_av_pack()

        latent_t = (length - 1) // 4 + 1
        latent_h = height // 16
        latent_w = width // 16

        t_sound = sound_token_count(length, fps)
        n_extra = extra_frames_fn(t_sound, latent_h, latent_w)
        n_samples = audio_sample_count(length, fps)

        packed = torch.zeros(
            [batch_size, 48, latent_t + n_extra, latent_h, latent_w],
            device=comfy.model_management.intermediate_device(),
        )

        latent = {
            "samples": packed,
            "cosmos3_sound_tokens": t_sound,
            "cosmos3_video_frames_lat": latent_t,
            "cosmos3_audio_samples": n_samples,
        }

        cond_meta = {"cosmos3_sound_tokens": t_sound}
        positive = node_helpers.conditioning_set_values(positive, cond_meta)
        negative = node_helpers.conditioning_set_values(negative, cond_meta)

        return (positive, negative, latent)


# ---------------------------------------------------------------------------
# 8. Cosmos3SplitAVLatent
# ---------------------------------------------------------------------------

class Cosmos3SplitAVLatent:
    """
    Splits a packed audio-video latent (output of the sampler) into separate
    video and audio latents.

    The latent must have been created by Cosmos3EmptyAVLatentVideo — the node
    reads cosmos3_video_frames_lat, cosmos3_sound_tokens, and
    cosmos3_audio_samples from the latent dict.

    At this point the samples are already in raw (un-normalized) space because
    process_latent_out has been applied by the sampler; no affine math is needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "COSMOS3_AUDIO_LATENT")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "split"
    CATEGORY = "Cosmos3"

    def split(self, latent):
        samples = latent["samples"]

        t_video_lat = latent.get("cosmos3_video_frames_lat")
        t_sound = latent.get("cosmos3_sound_tokens")
        n_samples = latent.get("cosmos3_audio_samples")

        if t_video_lat is None or t_sound is None or n_samples is None:
            raise ValueError(
                "latent was not created by Cosmos3EmptyAVLatentVideo — "
                "cosmos3_video_frames_lat, cosmos3_sound_tokens and/or "
                "cosmos3_audio_samples metadata keys are missing."
            )

        _sound_token_count, _extra_frames_fn, _audio_sample_count, unpack_sound = _get_av_pack()

        video_lat, sound_lat = unpack_sound(samples, int(t_video_lat), int(t_sound))

        video_latent = {"samples": video_lat}
        audio_latent = {
            "samples": sound_lat,
            "cosmos3_audio_samples": int(n_samples),
        }

        return (video_latent, audio_latent)


# ---------------------------------------------------------------------------
# 9. Cosmos3AudioDecode
# ---------------------------------------------------------------------------

class Cosmos3AudioDecode:
    """
    Decodes a COSMOS3_AUDIO_LATENT tensor (B, 64, T_s) to a ComfyUI AUDIO dict
    using the Cosmos3 Audio VAE decoder.

    The waveform is trimmed to the exact number of samples recorded in the
    latent dict (cosmos3_audio_samples) and returned as float32 on CPU at
    48 000 Hz stereo.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_vae": ("COSMOS3_AUDIO_VAE", {
                    "tooltip": "Audio VAE from Cosmos3 Loader (4th output).",
                }),
                "audio_latent": ("COSMOS3_AUDIO_LATENT", {
                    "tooltip": "Audio latent from Cosmos3 Split AV Latent.",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "Cosmos3"

    def decode(self, audio_vae, audio_latent):
        samples = audio_latent["samples"]
        n_samples = audio_latent.get("cosmos3_audio_samples")

        waveform = audio_vae.decode(samples)

        # Ensure float32 on CPU
        waveform = waveform.float().cpu()

        # Trim to exact waveform length if metadata is available
        if n_samples is not None:
            waveform = waveform[..., :int(n_samples)]

        return ({"waveform": waveform, "sample_rate": 48000},)


# ---------------------------------------------------------------------------
# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Cosmos3Loader": Cosmos3Loader,
    "Cosmos3TextEncode": Cosmos3TextEncode,
    "Cosmos3DefaultNegativePrompt": Cosmos3DefaultNegativePrompt,
    "Cosmos3EmptyLatentVideo": Cosmos3EmptyLatentVideo,
    "Cosmos3ImageToVideo": Cosmos3ImageToVideo,
    "Cosmos3Scheduler": Cosmos3Scheduler,
    "Cosmos3EmptyAVLatentVideo": Cosmos3EmptyAVLatentVideo,
    "Cosmos3SplitAVLatent": Cosmos3SplitAVLatent,
    "Cosmos3AudioDecode": Cosmos3AudioDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos3Loader": "Cosmos3 Loader",
    "Cosmos3TextEncode": "Cosmos3 Text Encode",
    "Cosmos3DefaultNegativePrompt": "Cosmos3 Default Negative Prompt",
    "Cosmos3EmptyLatentVideo": "Cosmos3 Empty Latent Video",
    "Cosmos3ImageToVideo": "Cosmos3 Image to Video",
    "Cosmos3Scheduler": "Cosmos3 Scheduler (UniPC sigmas)",
    "Cosmos3EmptyAVLatentVideo": "Cosmos3 Empty AV Latent Video",
    "Cosmos3SplitAVLatent": "Cosmos3 Split AV Latent",
    "Cosmos3AudioDecode": "Cosmos3 Audio Decode",
}
