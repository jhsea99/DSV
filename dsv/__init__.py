"""DSV: Context-Detached Self-Verification for object hallucination in LLaVA-1.5.

Kim, Hwang and Song, "Mitigating Object Hallucination in Vision-Language Models via Context-Detached Self-Verification"
(2026).

    from PIL import Image
    from dsv import DSV

    dsv = DSV.from_pretrained("liuhaotian/llava-v1.5-7b")
    image = Image.open("example.jpg").convert("RGB")
    out = dsv.caption(image)
    print(out["caption"])                          # DSV caption
    print(dsv.answer(image, ["dog", "frisbee"]))   # yes / no per object (the DSV verdict, as on POPE)
"""

from __future__ import annotations

from PIL import Image

from .decode import CaptionDecoder
from .locate import SimilarityLocator
from .model import LlavaRunner
from .samples import sample_captions
from .text import question, sample_words
from .verify import ImageVerifier

__all__ = ["DSV"]
__version__ = "1.0.0"


class DSV:
    """DSV with its fixed settings: K = 4 samples at temperature 1.2 (256 tokens), at most 3 rewrites per sentence,
    512 new tokens."""

    def __init__(self, runner: LlavaRunner, k: int = 4, temperature: float = 1.2, sample_tokens: int = 256,
                 max_new_tokens: int = 512, max_rewrites: int = 3):
        self.runner = runner
        self.k, self.temperature, self.sample_tokens = k, temperature, sample_tokens
        self.locator = SimilarityLocator(runner)
        self.decoder = CaptionDecoder(runner, self.locator, max_new_tokens=max_new_tokens, max_rewrites=max_rewrites)

    @classmethod
    def from_pretrained(cls, model_path: str = "liuhaotian/llava-v1.5-7b", vision_tower: str | None = None,
                        preprocess: str = "crop", device: str = "cuda", **kwargs) -> "DSV":
        return cls(LlavaRunner(model_path, vision_tower=vision_tower, preprocess=preprocess, device=device), **kwargs)

    def samples(self, image: Image.Image, seed: int = 0) -> list[str]:
        """The K sampled captions the verdict counts objects in."""
        return sample_captions(self.runner, image, k=self.k, temperature=self.temperature,
                               max_new_tokens=self.sample_tokens, seed=seed)

    def caption(self, image: Image.Image, samples: list[str] | None = None, seed: int = 0) -> dict:
        """A DSV caption. Keys: caption, sentences (every version with its objects, scores and rejections), verdicts
        (per object: m, c, m2, s, crop rect), stats, samples, n_tokens, seconds, and caption_withheld (an optional output
        not used in the paper: the caption without the sentences still rejected after the last rewrite)."""
        samples = samples if samples is not None else self.samples(image, seed=seed)
        out = self.decoder(image, sample_words(samples[: self.k]))
        out["samples"] = samples
        return out

    def answer(self, image: Image.Image, objects: list[str], samples: list[str] | None = None, seed: int = 0) -> dict:
        """The DSV verdict for each object ("Is there a/an {object} in the image?"): {object: {answer, s, m, c, m2,
        greedy}}; answer is "yes" when s >= 0. greedy is the plain LLaVA answer on the same logits."""
        samples = samples if samples is not None else self.samples(image, seed=seed)
        ver = ImageVerifier(self.runner, self.locator, image, sample_words(samples[: self.k]))
        out = {}
        for obj in objects:
            v = ver.verdict(obj.lower(), question(obj.lower()))
            out[obj] = {"answer": "no" if v["rejected"] else "yes", "s": v["s"], "m": v["m"], "c": v["c"], "m2": v["m2"],
                        "greedy": "yes" if v["greedy"] else "no"}
        return out
