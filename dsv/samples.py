"""K independent captions sampled at temperature 1.2 (top-k off, top-p 1), the evidence of the sample consensus."""

from __future__ import annotations

import torch
from PIL import Image

from .text import CAPTION_PROMPT


@torch.no_grad()
def sample_captions(runner, image: Image.Image, k: int = 4, temperature: float = 1.2, max_new_tokens: int = 256,
                    seed: int = 0) -> list[str]:
    prompt = runner.prompt_ids(CAPTION_PROMPT, suffix="").to(runner.device)
    pix = runner.pixel_values(image).unsqueeze(0).to(runner.device, torch.float16)
    torch.manual_seed(seed)
    g = runner.model.generate(inputs=prompt, images=pix, image_sizes=[image.size], use_cache=True, do_sample=True,
                              temperature=temperature, top_k=0, top_p=1.0, num_return_sequences=k,
                              max_new_tokens=max_new_tokens)
    return [s.strip() for s in runner.tok.batch_decode(g, skip_special_tokens=True)]
