# ComfyUI-Cosmos3

ComfyUI custom nodes for the [NVIDIA Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano) family of
Mixture-of-Transformers world models — text-to-video, image-to-video, and joint audio-video
generation. Cosmos3 runs language and diffusion tokens through one transformer, so there is no
separate text encoder.

Supported checkpoints:

| Checkpoint | Verified paths |
|---|---|
| [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) | text → video, image → video, joint audio-video |
| [`nvidia/Cosmos3-Super-Image2Video`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video) | image → video, text → video (no audio branch) |

The architecture is read from the checkpoint's `transformer/config.json`, so width, depth, base fps
and the presence of the audio branch all follow the model you point the loader at.

## Features

- Text-to-video, image-to-video (first-frame), and audio-video — all through the built-in
  KSampler / SamplerCustomAdvanced stack.
- Hooks into ComfyUI's dynamic VRAM, so the model streams from system RAM rather than
  needing to be resident in VRAM — see [VRAM](#vram).
- `fp8_e4m3fn` weights as a smaller, faster alternative to the default bf16.
- Flow-matching UniPC schedule (shifted, `uni_pc_bh2`).
- The text ("und") tower is prefilled once per prompt and its per-layer K/V reused across
  denoising steps.

## Install

1. Clone into `ComfyUI/custom_nodes/`.
2. `pip install -r requirements.txt` (just `transformers>=4.51`; torch/safetensors come with ComfyUI).
3. Download a checkpoint into `ComfyUI/models/cosmos3/<name>/` (or any path you pass to the
   loader). The loader expects these subfolders:

   ```
   <checkpoint>/
     transformer/    (sharded safetensors + index + config.json)
     vae/            (diffusion_pytorch_model.safetensors + config.json)
     text_tokenizer/ (tokenizer.json, vocab.json, merges.txt, ...)
     sound_tokenizer/(only on checkpoints with audio; enables the audio nodes)
     assets/         (optional — negative_prompt.json for the default-negative node)
   ```

   Shards are loaded sequentially, so loading requires room for the model plus one shard.

**Requirements:** ComfyUI with dynamic VRAM (`comfy-aimdo`) — developed and tested against
ComfyUI 0.24 · Python ≥ 3.10 · `transformers` ≥ 4.51 · enough system RAM to hold the checkpoint,
which is staged there and streamed to the GPU.

On ComfyUI builds without dynamic VRAM the model must be fully resident in VRAM; `fp8_e4m3fn` is
recommended there.

## Nodes

| Node | Purpose · key inputs → outputs |
|------|--------------------------------|
| **Cosmos3 Loader** | Load the model. `weight_dtype` = `default` (bf16) or `fp8_e4m3fn` (smaller, faster). → `MODEL`, `COSMOS3_TEXT_ENCODER`, `VAE`, `COSMOS3_AUDIO_VAE` (empty on checkpoints without audio) |
| **Cosmos3 Text Encode** | Tokenize a prompt (chat template + optional resolution/duration metadata). Set `width`/`height`/`num_frames`/`fps` to match the latent. → `CONDITIONING` |
| **Cosmos3 Default Negative Prompt** | Returns the checkpoint's bundled `assets/negative_prompt.json` as a STRING — wire into a negative Text Encode. Returns an empty string when the checkpoint ships no such file. → `STRING` |
| **Cosmos3 Empty Latent Video** | Zero latent `[B, 48, (length-1)//4+1, H/16, W/16]`. → `LATENT` |
| **Cosmos3 Image to Video** | First-frame conditioning: encodes the image and attaches it to positive/negative conds. → `CONDITIONING ×2`, `LATENT` |
| **Cosmos3 Scheduler** | Flow sigmas over [1, 0] with discrete shift. Output → `SamplerCustomAdvanced`. → `SIGMAS` |
| **Cosmos3 Empty AV Latent Video** | Packed audio+video latent; routes conds through to attach the sound-token count. → `CONDITIONING ×2`, `LATENT` |
| **Cosmos3 Split AV Latent** | Splits the denoised packed latent into video + audio latents. → `LATENT`, `COSMOS3_AUDIO_LATENT` |
| **Cosmos3 Audio Decode** | Decodes the audio latent to a 48 kHz stereo waveform. → `AUDIO` |

Sampling uses the built-in `CFGGuider` + `SamplerCustomAdvanced` with `KSamplerSelect` set to
**`uni_pc_bh2`**. `VAEDecode` → `CreateVideo` → `SaveVideo` for output.

The frame-count widgets accept up to **401** frames.

> **`flow_shift`** in the Scheduler is resolution-dependent — the reference inference uses
> **10 @ 720p, 5 @ 480p, 3 @ 256p**. `1.0` = no shift.

## Example workflows

Drag a JSON onto the canvas. All four share the same sampling setup — 832×480, 35 steps, cfg 6.0,
`uni_pc_bh2`, `flow_shift` 5.0 (the 480p value), bf16 weights — so they differ only in what is
being generated.

**`cosmos3_t2v.json` — text to video.** The baseline graph: prompt → `Cosmos3 Text Encode`, an
empty latent from `Cosmos3 Empty Latent Video`, sampled and decoded to 93 frames at 24 fps (3.9 s).
The negative prompt comes from `Cosmos3 Default Negative Prompt`, which supplies the checkpoint's
bundled negative prompt.

To change duration, set the frame count in all three places that carry it — both Text Encode nodes
and the latent node — otherwise the duration metadata disagrees with the latent and pacing is wrong.

**`cosmos3_i2v.json` — image to video.** Adds `LoadImage` → `Cosmos3 Image to Video`, which
encodes the image through the VAE, pins latent frame 0 to it and lets the rest denoise. That node
also emits the latent, so there is no separate empty-latent node. Swap `example.png` for your own
still; it is rescaled to `width`/`height`, so a matching aspect ratio avoids distortion.

**`cosmos3_t2v_audio.json` — joint audio and video.** Replaces the empty latent with
`Cosmos3 Empty AV Latent Video`, which packs sound tokens into extra latent frames so one sampler
pass denoises picture and sound together. The tail then splits: `Cosmos3 Split AV Latent` →
`Cosmos3 Audio Decode` for a 48 kHz stereo waveform, alongside the usual `VAEDecode` for frames.
Requires a checkpoint with a `sound_tokenizer` (Nano has one, Super does not). Note the AV node
takes its own `fps`, which must match Text Encode and `CreateVideo`.

**`cosmos3_super_i2v.json` — Cosmos3-Super image to video.** Points at
`Cosmos3-Super-Image2Video`, with every `fps` widget set to **16** rather than 24, because that is
Super's `base_fps`. 93 frames is 5.8 s at that rate. Super has no audio branch, so the loader's
audio output is unused. Super also ships no `assets/negative_prompt.json`, so
`Cosmos3 Default Negative Prompt` yields an empty string on it — type a negative prompt directly
into the negative Text Encode if you want one.

**For I2V + audio**, chain Text Encode → Image to Video → Empty AV Latent Video (use the AV node's
latent; discard the I2V one), then sample as usual.

## Recommended settings

35 steps · cfg 6.0 · sampler `uni_pc_bh2` · `flow_shift` 10/5/3 for 720p/480p/256p. Plain text
prompts work, and NVIDIA's JSON-upsampled prompts (cosmos-framework) can be pasted directly into
the prompt field.

Set `fps` to the checkpoint's `base_fps` — **24 for Cosmos3-Nano, 16 for Cosmos3-Super** — in Text
Encode, the AV latent node and `CreateVideo`. It feeds the duration metadata sentence and the
temporal position ids, so a mismatch shows up as wrong pacing. At 24 fps, 189 frames is 7.9 s.

## VRAM

With dynamic VRAM the weights stream from system RAM, so available VRAM bounds the *activations*
rather than the model, and checkpoints larger than the card can hold are usable. Frame count and
resolution, rather than checkpoint size, are what govern whether a given job fits.

Streaming moves weight data across the bus during sampling, so less available VRAM means more
transfer per step. `fp8_e4m3fn` reduces how much has to be streamed, and is not required in order
to fit.

## Limitations

- **Audio input** conditioning is unsupported — the checkpoints ship only the AVAE decoder.
- The three audio nodes need a checkpoint with a `sound_tokenizer`. Cosmos3-Super is video-only,
  so on it the loader's audio output is empty and those nodes do not apply.
- No action conditioning, reasoning/CoT, or super-resolution.

## Troubleshooting

- **OOM:** first check the ComfyUI log says `prepared for dynamic VRAM loading` — without it the
  whole model has to be resident. `--lowvram` does nothing when dynamic VRAM is on. Otherwise
  lower `length`/resolution or use `fp8_e4m3fn`.
- **Blurry output:** match `flow_shift` to resolution (10/5/3), try more steps, or a
  JSON-upsampled prompt.
- **Wrong duration / flicker:** make `num_frames` (Text Encode) match `length` (latent node), and
  keep both metadata templates on for positive and negative.

## License

MIT (node code) — see [LICENSE](LICENSE). Model weights are NVIDIA's, under OpenMDW-1.1.
