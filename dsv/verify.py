"""The DSV verdict on one image: two pieces of evidence, combined.

For an object X:
  m   context-detached question: the log-odds of Yes over No for "Is there a X in the image?", asked with the image in a
      new context without the caption (the system prompt and image are encoded once per image);
  c   sample consensus: how many of the K captions sampled at temperature 1.2 mention X; l = log((c + 1) / (K - c + 1));
  m2  crop question: when m + l < 0, the same question on a quarter-side crop around the point where the layer-8
      image-token states are most similar to "a photo of a X";
  s   = (max(m, m2) if m + l < 0 else m) + l. X is rejected when s < 0.
s >= 0 means the question's odds times the samples' odds is at least 1 (sigmoid(s) >= 1/2).
"""

from __future__ import annotations

import math

import torch
from PIL import Image

from .locate import square_crop
from .text import forms, head, question


class ImageVerifier:
    def __init__(self, runner, locator, image: Image.Image, sample_sets: list[set[str]]):
        self.r, self.loc, self.image, self.samples = runner, locator, image, sample_sets
        self.verdicts: dict[str, dict] = {}
        self.stats = {"questions": 0, "crops": 0}
        pix = runner.pixel_values(image).unsqueeze(0).to(runner.device, torch.float16)
        with torch.no_grad():
            op = runner.model(input_ids=runner.prefix_ids, images=pix, image_sizes=[image.size], use_cache=True,
                              output_hidden_states=True)
        self.prefix_pkv, self.prefix_hs = op.past_key_values, op.hidden_states

    @torch.no_grad()
    def ask_prefix(self, q: str) -> tuple[float, bool]:
        """Margin of the question on the cached system + image prefix."""
        ids = self.r.prompt_ids(q)[0].tolist()
        assert ids[: self.r.prefix_len] == self.r.prefix_ids[0].tolist(), "question does not share the cached prefix"
        oq = self.r.model(input_ids=torch.tensor([ids[self.r.prefix_len:]], device=self.r.device),
                          past_key_values=self.prefix_pkv, use_cache=True, output_hidden_states=True)
        return self.r.margin(oq.hidden_states[-1][0, -1])

    @torch.no_grad()
    def ask(self, img: Image.Image, q: str):
        """Margin, forward output and image-token span of the question on an image (a crop)."""
        ids = self.r.prompt_ids(q)
        start, stop = self.r.image_positions(ids)
        o = self.r.forward(ids, img)
        return self.r.margin(o.hidden_states[-1][0, -1])[0], o, start, stop

    def second_view(self, phrase: str, q: str):
        """m2 on the quarter-side crop: a half-side square at the full view's similarity peak, then a half-side square of
        that crop at the crop's own peak. Returns (m2, rect, peak) in image pixels."""
        img = self.image
        W, H = img.size
        text = f"a photo of {'an' if phrase[0] in 'aeiou' else 'a'} {phrase}"
        start, stop = self.r.prefix_span
        peak, _ = self.loc.peak(self.prefix_hs, start, stop, text, W, H)
        rect1 = square_crop(peak, W, H, max(W, H) / 2)
        crop1 = img.crop(tuple(rect1))
        w1, h1 = crop1.size
        _, o1, s1, e1 = self.ask(crop1, q)
        pk, _ = self.loc.peak(o1.hidden_states, s1, e1, text, w1, h1)
        del o1
        r2 = square_crop(pk, w1, h1, max(w1, h1) / 2)
        rect = [rect1[0] + r2[0], rect1[1] + r2[1], rect1[0] + r2[2], rect1[1] + r2[3]]
        m2, _, _, _ = self.ask(img.crop(tuple(rect)), q)
        return m2, rect, (rect1[0] + pk[0], rect1[1] + pk[1])

    def consensus(self, phrase: str) -> tuple[int, float]:
        c = sum(bool(forms(head(phrase)) & sw) for sw in self.samples)
        return c, math.log((c + 1) / (len(self.samples) - c + 1))

    def verdict(self, phrase: str, q: str | None = None) -> dict:
        """{m, greedy, c, lo, m2, rect, peak, s, rejected}; cached per image."""
        if phrase not in self.verdicts:
            q = q or question(phrase)
            m, greedy = self.ask_prefix(q)
            self.stats["questions"] += 1
            c, lo = self.consensus(phrase)
            v = {"m": m, "greedy": greedy, "c": c, "lo": lo, "m2": None, "rect": None, "peak": None}
            if m + lo < 0:
                m2, rect, peak = self.second_view(phrase, q)
                self.stats["crops"] += 2
                v.update({"m2": m2, "rect": rect, "peak": [round(peak[0], 1), round(peak[1], 1)]})
            v["s"] = (m if v["m2"] is None else max(m, v["m2"])) + lo
            v["rejected"] = v["s"] < 0
            self.verdicts[phrase] = v
        return self.verdicts[phrase]
