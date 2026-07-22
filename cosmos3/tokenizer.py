"""
cosmos3/tokenizer.py

Cosmos3TextEncoderWrapper — tokenizer-only object for Cosmos3.

Ports tokenize_prompt from the diffusers Cosmos3OmniPipeline:
- optional metadata sentences (duration + resolution templates)
- apply_chat_template (Qwen2TokenizerFast), user turn only by default
  (use_system_prompt=False; overridable per call)
- add_generation_prompt=True, add_vision_id=False
- append [eos_token_id, vision_start_id]
"""

import json
import os
from typing import List, Optional

# transformers is a required dependency (see requirements.txt)
from transformers import AutoTokenizer


_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."

# Metadata sentence templates (from pipeline_cosmos3_omni.py)
_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_VIDEO_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."
_INVERSE_DURATION_TEMPLATE = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
_INVERSE_VIDEO_RESOLUTION_TEMPLATE = "This video is not of {height}x{width} resolution."
_INVERSE_IMAGE_RESOLUTION_TEMPLATE = "This image is not of {height}x{width} resolution."


class Cosmos3TextEncoderWrapper:
    """
    Wraps the Qwen2 tokenizer for Cosmos3 text conditioning.

    Does NOT run a text encoder — token IDs go directly into the transformer.
    The transformer's embed_tokens (nn.Embedding) handles embedding internally.
    """

    def __init__(self, model_dir: str, tokenizer=None):
        """
        Args:
            model_dir: Path to the cosmos3-nano model root directory.
            tokenizer: Pre-loaded AutoTokenizer instance (optional; loaded from
                       <model_dir>/text_tokenizer if not provided).
        """
        self.model_dir = model_dir
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            tokenizer_path = os.path.join(model_dir, "text_tokenizer")
            if not os.path.isdir(tokenizer_path):
                # Edge ships its (Nemotron) tokenizer at the checkpoint root rather
                # than in a text_tokenizer/ subfolder.
                tokenizer_path = model_dir
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Resolve special token IDs
        self._eos_token_id = self.tokenizer.eos_token_id
        self._vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(
        self,
        prompt: str,
        width: int = 1280,
        height: int = 720,
        num_frames: int = 189,
        fps: float = 24.0,
        add_resolution_template: bool = True,
        add_duration_template: bool = True,
        is_negative: bool = False,
        use_system_prompt: bool = False,
    ) -> List[int]:
        """
        Tokenize a prompt using the Cosmos3 chat template.

        Faithfully ports tokenize_prompt from pipeline_cosmos3_omni.py:
          1. Append metadata sentences (duration + resolution).
          2. Apply Qwen2 chat template with add_generation_prompt=True,
             add_vision_id=False.
          3. Append [eos_token_id, <|vision_start|>_id].

        Args:
            prompt: The text prompt.
            width: Output video width in pixels.
            height: Output video height in pixels.
            num_frames: Number of video frames.
            fps: Frames per second.
            add_resolution_template: Whether to append resolution sentence.
            add_duration_template: Whether to append duration sentence.
            is_negative: Whether this is a negative prompt (uses inverse templates).
            use_system_prompt: Whether to prepend a system turn (default False).

        Returns:
            List of integer token IDs.
        """
        is_image = (num_frames == 1)

        # Select templates
        if is_image:
            res_template = _IMAGE_RESOLUTION_TEMPLATE
            inv_res_template = _INVERSE_IMAGE_RESOLUTION_TEMPLATE
        else:
            res_template = _VIDEO_RESOLUTION_TEMPLATE
            inv_res_template = _INVERSE_VIDEO_RESOLUTION_TEMPLATE

        # Append metadata sentences
        text = prompt

        def _append(base: str, addition: str) -> str:
            base = base.rstrip(".")
            return f"{base}. {addition}" if base else addition

        if not is_image and add_duration_template:
            duration = num_frames / fps
            if is_negative:
                text = _append(text, _INVERSE_DURATION_TEMPLATE.format(duration=duration, fps=fps))
            else:
                text = _append(text, _DURATION_TEMPLATE.format(duration=duration, fps=fps))

        if add_resolution_template:
            if is_negative:
                text = _append(text, inv_res_template.format(height=height, width=width))
            else:
                text = _append(text, res_template.format(height=height, width=width))

        # Build chat conversation
        conversations = []
        if use_system_prompt:
            system_prompt = _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO
            conversations.append({"role": "system", "content": system_prompt})
        conversations.append({"role": "user", "content": text})

        # Apply chat template
        result = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=True,
        )

        # Append special tokens: [eos_token_id, <|vision_start|>]
        input_ids = list(result.input_ids) + [
            self._eos_token_id,
            self._vision_start_id,
        ]

        return input_ids

    def default_negative_prompt(self) -> str:
        """
        Returns the bundled negative prompt string.

        Reads <model_dir>/assets/negative_prompt.json and returns json.dumps of it.
        Returns "" if the file does not exist.
        """
        neg_path = os.path.join(self.model_dir, "assets", "negative_prompt.json")
        if os.path.exists(neg_path):
            with open(neg_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return json.dumps(obj)
        return ""
