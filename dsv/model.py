"""LLaVA-1.5 on the official LLaVA code path (github.com/haotian-liu/LLaVA): loading, prompts, image preprocessing and the
yes/no margin of a question."""

from __future__ import annotations

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import expand2square, tokenizer_image_token
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from transformers import AutoTokenizer

from .text import POPE_SUFFIX

GRID = 24  # 336 px / patch 14
N_IMG = GRID * GRID
YES_WORDS = ("Yes", "yes", "YES")
NO_WORDS = ("No", "no", "NO")
PREPROCESS = ("crop", "pad")


class LlavaRunner:
    """LLaVA-1.5 in fp16 with eager attention.

    preprocess: "crop" resizes the shortest side to 336 and centre-crops (CLIPImageProcessor.preprocess, the preprocessing
    of the VCD, OPERA, PAI and SumGD evaluation code; the paper's main tables); "pad" pads to a square with the CLIP mean
    colour first (LLaVA-1.5's own setting, image_aspect_ratio "pad")."""

    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-7b", vision_tower: str | None = None,
                 preprocess: str = "crop", device: str = "cuda"):
        if preprocess not in PREPROCESS:
            raise ValueError(f"preprocess must be one of {PREPROCESS}")
        self.preprocess = preprocess
        cfg = LlavaConfig.from_pretrained(model_path)
        if vision_tower:
            cfg.mm_vision_tower = vision_tower
        cfg._attn_implementation = "eager"  # an explicit config keeps its own setting, so set it here, not only as a kwarg
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path, config=cfg, torch_dtype=torch.float16,
                                                           low_cpu_mem_usage=True, attn_implementation="eager")
        vt = self.model.get_vision_tower()
        if not vt.is_loaded:
            vt.load_model()
        self.model.to(device).eval()
        vt.to(device=device, dtype=torch.float16)
        self.image_processor = vt.image_processor
        self.device = self.model.device
        self.lm_head32 = self.model.lm_head.weight.float()
        self.yes_ids = torch.tensor(self._variant_ids(YES_WORDS), device=self.device)
        self.no_ids = torch.tensor(self._variant_ids(NO_WORDS), device=self.device)
        # every question shares the system prompt and the image: that prefix is encoded once per image
        q_a = self.prompt_ids("Is there a cat in the image?")[0].tolist()
        q_b = self.prompt_ids("Are there dogs in the image?")[0].tolist()
        self.prefix_len = next(i for i, (a, b) in enumerate(zip(q_a, q_b)) if a != b)
        self.prefix_ids = torch.tensor([q_a[: self.prefix_len]], device=self.device)
        self.prefix_span = self.image_positions(self.prefix_ids)

    def _variant_ids(self, words) -> list[int]:
        """Vocabulary ids of each word as a SentencePiece piece, with and without the leading-space marker."""
        ids = set()
        for w in words:
            for piece in ("▁" + w, w):
                i = self.tok.convert_tokens_to_ids(piece)
                if i is not None and i != self.tok.unk_token_id:
                    ids.add(int(i))
        return sorted(ids)

    def prompt_ids(self, text: str, with_image: bool = True, suffix: str = POPE_SUFFIX) -> torch.Tensor:
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], (DEFAULT_IMAGE_TOKEN + "\n" if with_image else "") + text + suffix)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if with_image:
            return tokenizer_image_token(prompt, self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        return self.tok(prompt, return_tensors="pt").input_ids

    def pixel_values(self, image: Image.Image) -> torch.Tensor:
        if self.preprocess == "crop":
            return self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        square = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        return self.image_processor.preprocess(square, return_tensors="pt")["pixel_values"][0]

    @staticmethod
    def image_positions(input_ids: torch.Tensor) -> tuple[int, int]:
        """[start, stop) of the image features in the expanded sequence (the placeholder becomes N_IMG tokens)."""
        p = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
        assert len(p) == 1, "expected exactly one image placeholder"
        start = int(p[0])
        return start, start + N_IMG

    def forward(self, input_ids: torch.Tensor, image: Image.Image | None = None):
        """One forward without cache; hidden states always."""
        pixels = None if image is None else self.pixel_values(image)
        return self.model(input_ids=input_ids.to(self.device),
                          images=None if pixels is None else pixels.unsqueeze(0).to(self.device, torch.float16),
                          image_sizes=None if pixels is None else [image.size],
                          output_hidden_states=True, output_attentions=False, use_cache=False, return_dict=True)

    def margin(self, last_hidden: torch.Tensor) -> tuple[float, bool]:
        """(log-odds of yes over no, official greedy yes) at the answer position, in fp32."""
        lg = self.lm_head32 @ last_hidden.float()
        m = float(torch.logsumexp(lg[self.yes_ids], 0) - torch.logsumexp(lg[self.no_ids], 0))
        return m, bool(lg[self.yes_ids].max() > lg[self.no_ids].max())
