"""
Split a Cosmos3 transformer checkpoint into its two towers.

Cosmos3 is a Mixture-of-Transformers: the understanding ("reasoner") and
generation pathways run through the same layer stack but use entirely disjoint
weights — measured on a 36-layer bf16 checkpoint, 397 tensors and 14.10 GB on
each side. The und tower is a text encoder in all but name: causal over the
prompt alone, invariant across denoising steps, needed exactly once.

Splitting on disk is optional: ComfyUI-Cosmos3 drops the unwanted half while
streaming the original shards, so an unmodified checkpoint loads either half.
Split when the download and the bytes on disk should shrink too, or to publish
the reasoner as its own file.

    python split_cosmos3_checkpoint.py <checkpoint_dir> <output_dir>

Produces <output_dir>/und/transformer/ and <output_dir>/gen/transformer/, each
loadable by pointing the matching node at it. Pre-quantized checkpoints (int8,
int4/ConvRot) split unchanged — the per-layer quantization metadata rides along
with the tensors it belongs to.
"""

import argparse
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

# Share the runtime's classification rather than restating it; the two
# disagreeing would produce a split that loads with missing keys.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cosmos3.towers import is_dropped_key, is_und_key, resolve_shard_files  # noqa: E402

# Target bytes per output shard, matching the upstream checkpoint layout.
SHARD_LIMIT = 5 * 1000 ** 3


def _write_half(tensors, out_dir, name):
    """Write {key: tensor} as one or more shards plus an index, if sharded."""
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted(tensors)
    total = sum(tensors[k].numel() * tensors[k].element_size() for k in keys)

    if total <= SHARD_LIMIT:
        save_file(tensors, os.path.join(out_dir, "diffusion_pytorch_model.safetensors"),
                  metadata={"format": "pt"})
        print(f"  {name}: 1 shard, {total / 1024 ** 3:.2f} GB, {len(keys)} tensors")
        return

    shards, current, current_bytes = [], {}, 0
    for k in keys:
        nbytes = tensors[k].numel() * tensors[k].element_size()
        if current and current_bytes + nbytes > SHARD_LIMIT:
            shards.append(current)
            current, current_bytes = {}, 0
        current[k] = tensors[k]
        current_bytes += nbytes
    if current:
        shards.append(current)

    weight_map = {}
    for i, shard in enumerate(shards, 1):
        fname = f"diffusion_pytorch_model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(out_dir, fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(out_dir, "diffusion_pytorch_model.safetensors.index.json"),
              "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"  {name}: {len(shards)} shards, {total / 1024 ** 3:.2f} GB, {len(keys)} tensors")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_dir", help="Checkpoint root (contains transformer/)")
    ap.add_argument("output_dir", help="Where to write und/ and gen/")
    ap.add_argument("--half", choices=["und", "gen", "both"], default="both",
                    help="Write only one half (default: both)")
    args = ap.parse_args()

    src = os.path.join(args.checkpoint_dir, "transformer")
    config = os.path.join(src, "config.json")
    if not os.path.exists(config):
        raise SystemExit(f"No transformer/config.json under {args.checkpoint_dir}")

    shard_paths = resolve_shard_files(src)
    print(f"Reading {len(shard_paths)} shard(s) from {src}")

    und, gen, dropped_bytes = {}, {}, 0
    for path in shard_paths:
        print(f"  {os.path.basename(path)}")
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if is_dropped_key(key):
                    dropped_bytes += tensor.numel() * tensor.element_size()
                    continue
                (und if is_und_key(key) else gen)[key] = tensor

    print(f"Dropped {dropped_bytes / 1024 ** 3:.2f} GB (lm_head / action head / rope buffers)")

    for half, tensors in (("und", und), ("gen", gen)):
        if args.half not in (half, "both"):
            continue
        out = os.path.join(args.output_dir, half, "transformer")
        _write_half(tensors, out, half)
        shutil.copy2(config, os.path.join(out, "config.json"))

    print("\nDone. Point Cosmos3 Und Tower Loader at "
          f"{os.path.join(args.output_dir, 'und')} and Cosmos3 Loader "
          f"(split_reasoner enabled) at {os.path.join(args.output_dir, 'gen')}.")


if __name__ == "__main__":
    main()
