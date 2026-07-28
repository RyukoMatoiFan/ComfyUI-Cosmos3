# ComfyUI-Cosmos3

ComfyUI custom nodes for the [NVIDIA Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano) family of
Mixture-of-Transformers world models — text-to-video, image-to-video, and joint audio-video
generation. Cosmos3 runs language and diffusion tokens through one transformer, so there is no
separate text encoder.

Supported checkpoints:

| Checkpoint | Verified paths |
|---|---|
| [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) | text → video, image → video, joint audio-video |
| [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) | text → video, image → video, joint audio-video |
| [`nvidia/Cosmos3-Super-Image2Video`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video) | image → video, text → video (no audio branch) |
| [`nvidia/Cosmos3-Super-Image2Video-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video-4Step) | image → video (DMD2-distilled, 4 steps, cfg 1) |
| [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge) | text → video, image → video (Nemotron-dense backbone) |

The architecture is read from the checkpoint's `transformer/config.json`, so width, depth, base fps
and the presence of the audio branch all follow the model you point the loader at.

## Features

- Text-to-video, image-to-video (first-frame), and audio-video — all through the built-in
  KSampler / SamplerCustomAdvanced stack.
- Hooks into ComfyUI's dynamic VRAM, so the model streams from system RAM rather than
  needing to be resident in VRAM — see [VRAM](#vram).
- `fp8_e4m3fn` weights as a smaller alternative to the default bf16 (less data to stream).
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
| **Cosmos3 Loader** | Load the model. `weight_dtype` = `default` (bf16), `fp8_e4m3fn`, or `int8` (on-the-fly ComfyUI-native int8). Pre-quantized checkpoints (see [Quantized weights](#quantized-weights)) are auto-detected from their metadata — load those with `default`. `split_reasoner` loads only the generator half (see [Splitting the reasoner](#splitting-the-reasoner)). → `MODEL`, `COSMOS3_TEXT_ENCODER`, `VAE`, `COSMOS3_AUDIO_VAE` (empty on checkpoints without audio) |
| **Cosmos3 Und Tower Loader** | Optional. Loads only the understanding tower ("reasoner") from the same `model_dir`, so it can be prefilled once and unloaded. → `COSMOS3_UND_TOWER` |
| **Cosmos3 Text Encode** | Tokenize a prompt (chat template + optional resolution/duration metadata). Set `width`/`height`/`num_frames`/`fps` to match the latent. Connect `und_tower` to run the reasoner here instead of inside the model. → `CONDITIONING` |
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

Drag a JSON onto the canvas. All graphs except the 4-step one share the same sampling setup —
832×480, 35 steps, cfg 6.0, `uni_pc_bh2`, `flow_shift` 5.0 (the 480p value) — and differ only in what
is generated; `cosmos3_super_i2v_4step.json` uses its own 4-step schedule (see below). They also work
with the [quantized checkpoints](#quantized-weights): point the loader at a folder whose
`transformer/` holds a quantized file and it loads from the metadata.

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

**`cosmos3_super_i2v_4step.json` — Cosmos3-Super-Image2Video-4Step (distilled).** The DMD2-distilled
4-step variant. Same graph as `cosmos3_super_i2v.json` but: `Cosmos3 Scheduler` set to
`schedule = distilled_4step` (its fixed 4-step schedule, so `steps`/`flow_shift` are ignored), the
sampler switched to **euler**, and `CFGGuider` cfg **1.0** (the model is guidance-distilled, so no
CFG). fps stays 16. Four steps instead of 35.

**`cosmos3_super_i2v_split.json` — reasoner loaded separately.** `cosmos3_super_i2v.json` with
`split_reasoner` enabled on the loader and a `Cosmos3 Und Tower Loader` wired into the `und_tower`
input of both Text Encode nodes. Both loaders take the same `Cosmos3-Super-Image2Video` path; the
split is a load-time option, not a different checkpoint. See
[Splitting the reasoner](#splitting-the-reasoner). The other examples do not use it.

**`cosmos3_edge_t2v.json` — Cosmos3-Edge text to video.** Same graph as `cosmos3_t2v.json` (fps 24)
pointed at `Cosmos3-Edge`. Edge is a different backbone (`cosmos3_edge_nemotron_dense`: squared-ReLU
non-gated MLP, no text QK-norm, its own Nemotron tokenizer) — the loader reads all of this from
`transformer/config.json`, so no extra setup is needed. It is a smaller model with lower output
fidelity than Nano/Super.

**For I2V + audio**, chain Text Encode → Image to Video → Empty AV Latent Video (use the AV node's
latent; discard the I2V one), then sample as usual.

## Recommended settings

35 steps · cfg 6.0 · sampler `uni_pc_bh2` · `flow_shift` 10/5/3 for 720p/480p/256p. Plain text
prompts work, and NVIDIA's JSON-upsampled prompts (cosmos-framework) can be pasted directly into
the prompt field.

Set `fps` to the checkpoint's `base_fps` — **24 for Cosmos3-Nano, Cosmos3-Super and Cosmos3-Edge; 16 for both Cosmos3-Super-Image2Video variants (incl. -4Step)** — in Text
Encode, the AV latent node and `CreateVideo`. It feeds the duration metadata sentence and the
temporal position ids, so a mismatch shows up as wrong pacing. At 24 fps, 189 frames is 7.9 s.

## VRAM

With dynamic VRAM the weights stream from system RAM, so available VRAM bounds the *activations*
rather than the model, and checkpoints larger than the card can hold are usable. Frame count and
resolution, rather than checkpoint size, are what govern whether a given job fits.

Streaming moves weight data across the bus during sampling, so less available VRAM means more
transfer per step. `fp8_e4m3fn` reduces how much has to be streamed, and is not required in order
to fit.

## Splitting the reasoner

Cosmos3 is a Mixture-of-Transformers: the understanding (und) and generation pathways run through the
same layer stack but use disjoint weights. Per layer the und side is `self_attn.{to_q,to_k,to_v,
to_out,norm_q,norm_k,k_norm_und_for_gen}`, `mlp`, `input_layernorm` and `post_attention_layernorm`;
the gen side is the `add_*_proj`/`to_add_out` twins, `mlp_moe_gen` and the `*_moe_gen` norms. Top
level, `embed_tokens` and `norm` are und; `proj_in`, `proj_out`, `time_embedder`, `norm_moe_gen` and
the audio branch are gen.

Tensor sizes from a 36-layer, hidden 4096 bf16 checkpoint:

| | tensors | size | share of loaded weights |
|---|---|---|---|
| und | 398 | 14.10 GB | 52.1 % |
| gen | 410 | 12.97 GB | 47.9 % |
| `lm_head`, action head (never loaded) | 6 | 1.19 GB | — |

`forward_und` is causal over the text tokens and takes neither the timestep nor the latent, so its
per-layer K/V are constant across denoising steps, and `forward_gen` reads the und side only through
those K/V. The bundled model exploits this too — `_und_prefill` runs once per prompt and the K/V are
replayed — but with both pathways in one `nn.Module` the und tensors stay in the patcher's footprint
for the whole sampling run.

`split_reasoner` on **Cosmos3 Loader** constructs the transformer with `tower="gen"`, omitting the
und parameters. **Cosmos3 Und Tower Loader** constructs `tower="und"` as a separate `ModelPatcher`;
**Cosmos3 Text Encode** loads it, calls `prefill_und_packed`, and puts the result in the conditioning
as `cosmos3_und_kv` — shape `[1, num_layers, 2, L_text, H_kv, head_dim]`. Being two patchers, the
reasoner is eligible for eviction once the sampler requests the denoiser.

Both loaders read an unmodified checkpoint: `_filter_transformer_keys` drops the other half per shard
during streaming, so no separate file is required. `tools/split_cosmos3_checkpoint.py` writes `und/`
and `gen/` trees if you also want the disk and download halved; it imports the same classifier
(`cosmos3/towers.py`) the loader uses, and carries per-layer quantization metadata with its tensors,
so int8 and int4/ConvRot checkpoints split unchanged.

Where the reduction lands depends on the loading mode. Under streaming, weights sit in host RAM and
page to VRAM per forward, so min VRAM is activation-bound and unchanged; the saving is in host RAM.
With weights held on-card it is VRAM instead. Step time is unchanged either way.

Costs: the conditioning carries `num_layers × 2 × L_text × H_kv × head_dim` elements on the sampling
device (36 × 2 × 300 × 8 × 128 × 2 B ≈ 44 MB for a 300-token prompt at the shape above), one set per
prompt, so CFG holds two. A LoRA patching und-side keys must be applied to the und tower's patcher;
the `patches_uuid` cache invalidation in `model.py` covers only the bundled path.

Verified on H100 against a 36-layer, hidden 4096 checkpoint, 832×480×93, 35 steps, `uni_pc_bh2`.
Both halves load with no missing keys in every weight format tried, and the reasoner's per-layer K/V
are bitwise identical whether produced by the bundled model or by a separately-loaded und tower.

The denoised latent is **bitwise identical** between the bundled and split paths in bf16, int8 and
int4/ConvRot alike (`torch.equal` True in all three).

Quantized weights make this sharper than it sounds. They are tensor subclasses, and their dispatch
differs between `torch.inference_mode()` and `torch.no_grad()` — under one the ConvRot Hadamard
rotation is undone before the matmul, under the other it is not. Sampling runs under
`inference_mode`, so `Cosmos3 Text Encode` prefills under it too. Prefilling under `no_grad` instead
silently yields int4 K/V that do not match what the denoiser sees, with bf16 and int8 unaffected
because only ConvRot carries a rotation.

The partition covers the Nano/Super and Edge backbones, with and without the audio branch.

## Quantized weights

Pre-quantized transformers for the official `transformer/` are hosted at
[`AkaneTendo25/Cosmos3-ConvRot`](https://huggingface.co/AkaneTendo25/Cosmos3-ConvRot).

| Format | Status | What it is | Nano | Super |
|--------|--------|------------|------|-------|
| int8 | available | weight-only int8, per-row scale + Hadamard (ConvRot) | 16.5 GB | 65.7 GB |
| int4 | available (all models) | MLP int4 (GPTQ-calibrated + ConvRot Hadamard, AWQ W4A16 packing) + attention int8 | 12.4 GB | 46.8 GB |

Both Super-Image2Video variants (`Cosmos3-Super-Image2Video` and `-4Step`) are also available in
int4 at **46.7 GB** each (int8 is 65.6 GB each).

Both are weight-only — activations stay bf16, so the effect is lower memory use, not faster
inference (int4 is a little slower: it dequantizes per forward). They are lossy relative to bf16;
int4 has larger quantization error than int8, so at a fixed seed its output diverges from the bf16
result more than int8 does. Compare against the bf16 checkpoint for your use case.

The int4 MLPs are GPTQ-calibrated on captured activations (plain round-to-nearest int4, even with the
rotation, is too lossy on these two-tower models); the full step-by-step recipe with parameters is in
the [weights card](https://huggingface.co/AkaneTendo25/Cosmos3-ConvRot#how-these-were-quantized).

### Measured footprint

H100, 832×480, 93 frames. Measured at the *minimum* VRAM the clip runs in (maximum streaming). **Min VRAM** is that floor — it
is set by the activation working set (resolution × frame count), so it barely changes with the weight
format. **RAM** is the peak host memory the process uses at that point. **Time** is sampling + VAE
decode at the step count shown.

| Model | Format | RAM | Min VRAM | Time | Weights: bundled → split (und / gen) |
|-------|--------|-----|----------|------|------|
| Edge — t2v, 35 steps | bf16 | 14 GB | **≈6 GB** | 18 s | 5.76 → 3.13 / 2.64 GB |
| | int8 | 7 GB | | 25 s | 3.14 → 1.81 / 1.32 GB |
| | int4 | 7 GB | | 17 s | 2.40 → 1.44 / 0.95 GB |
| Nano — t2v, 35 steps | bf16 | 58 GB | **≈7 GB** | 47 s | 27.07 → 14.10 / 12.98 GB |
| | int8 | 21 GB | | 50 s | 14.15 → 7.63 / 6.51 GB |
| | int4 | 20 GB | | 49 s | 10.34 → 5.73 / 4.61 GB |
| Super — t2v, 35 steps | bf16 | 240 GB | **≈8 GB** | 168 s | 117.76 → 59.58 / 58.18 GB |
| | int8 | 67 GB | | 174 s | 59.67 → 30.53 / 29.14 GB |
| | int4 | 63 GB | | 168 s | 42.06 → 21.73 / 20.33 GB |
| Super-Image2Video — i2v, 35 steps | bf16 | 240 GB | **≈9 GB** | 167 s | 117.76 → 59.58 / 58.18 GB |
| | int8 | 67 GB | | 168 s | 59.67 → 30.53 / 29.14 GB |
| | int4 | 63 GB | | 165 s | 42.06 → 21.73 / 20.33 GB |
| Super-Image2Video-4Step — i2v, 4 steps | bf16 | 240 GB | **≈9 GB** | 42 s | 117.76 → 59.58 / 58.18 GB |
| | int8 | 67 GB | | 20 s | 59.67 → 30.53 / 29.14 GB |
| | int4 | 63 GB | | 16 s | 42.06 → 21.73 / 20.33 GB |

The table is with the reasoner bundled (the default). With
[`split_reasoner`](#splitting-the-reasoner) the und parameters are never constructed in the sampled
model, and peak host memory is set by whichever half is larger — the und tower, at 52.1 % of the
weights — instead of by the sum.

The last column is weight bytes, read from each checkpoint's tensor headers, not a runtime figure
like RAM: the loaded total, then what each half holds under
[`split_reasoner`](#splitting-the-reasoner). The denoiser holds the **gen** half for the whole run;
the **und** half is loaded first and released, so it — the larger of the two, 51–60 % depending on
model and format — is what bounds peak memory. RAM and Min VRAM are bundled-path measurements and
are not re-stated for the split, which was measured with a different instrument. Nano, same clip and
step count, through the nodes:

| Format | Sampled model | Peak VRAM | Time |
|---|---|---|---|
| bf16 — bundled | 27.07 GB | 33.3 GB | 40.0 s |
| bf16 — split | 12.98 GB | 19.3 GB | 36.6 s |
| int8 — bundled | 14.15 GB | 20.4 GB | 49.3 s |
| int8 — split | 6.51 GB | 12.8 GB | 45.1 s |
| int4 — bundled | 10.34 GB | 16.6 GB | 47.6 s |
| int4 — split | 4.61 GB | 10.9 GB | 44.5 s |

**Sampled model** is what the denoiser holds: the gen half alone under split, so it drops by the und
share. The und tower (5.7 GB int4 to 14.1 GB bf16) is loaded before it and released, which is why
peak, not the sampled size, is what a host has to accommodate. Peak VRAM falls too, because a
smaller model streams less. Sampling is a little faster for the same reason, though the prefill
itself runs once per prompt in both modes. Super was not part of this run.

RAM and VRAM trade off: these are at the minimum VRAM (maximum streaming); giving the GPU more VRAM
holds more weights on-card, which lowers the RAM figure and speeds sampling. bf16 host RAM peaks near
twice the weight size (staging), so the Super family is impractical in bf16 without a very large host.
For that family the int4 checkpoints fit a **64 GB** host (≈63 GB), while int8 needs a little more
(≈67 GB); Nano and Edge fit comfortably in any format.

Weight-only quantization lowers RAM and download size, not compute time: int8 is no faster (slightly
slower from the per-forward dequant), int4 ≈ bf16. The 4-step distilled schedule is the only real
speedup (4 steps vs 35).

The int4 format is W4A16 (int4 weights, bf16 activations). No GPU — Hopper or Blackwell — has an
int4×bf16 tensor-core instruction, so W4A16 always runs as dequantize-to-bf16 then a bf16 matmul; the
saving is memory/download only, with no compute speedup on any hardware. Blackwell's FP4 tensor cores
accelerate NVFP4/MXFP4 (4-bit weights *and* activations), a different scheme not used here.

**To use:** download the `*.safetensors` for a model, put it in `<checkpoint>/transformer/` renamed to
`diffusion_pytorch_model.safetensors` (remove the bf16 shards and `*.index.json`), and keep the
official `vae/`, tokenizers and `config.json`. The loader reads the format from the checkpoint
metadata — load with `weight_dtype = default`; no workflow or node change is needed. Requires a
ComfyUI build with comfy-kitchen (int8 from ≥ 0.27; int4 uses its AWQ W4A16 layout, dequantized
weight-only with the ConvRot rotation undone at load, so the quantized matmul is deterministic).

**Prompting Cosmos3-Edge:** Edge is trained on JSON-structured prompts and is less robust to plain
text than the larger Nano/Super. Plain text usually works, but on some detailed scenes (notably
reflective surfaces) it can produce flare/pulsation artifacts; wrapping the text as
`{"temporal_caption": "<your prompt>"}` avoids them. This is a base-model property, independent of
quantization (it shows in bf16 too).

## Limitations

- **Audio input** conditioning is unsupported — the checkpoints ship only the AVAE decoder.
- The three audio nodes need a checkpoint with a `sound_tokenizer`. Cosmos3-Super-Image2Video is
  video-only, so on it the loader's audio output is empty and those nodes do not apply.
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
