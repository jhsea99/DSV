"""Where to look again: the peak of the layer-8 similarity map between the image tokens and a phrase, and the square crop
around it. No coordinates are generated and no attention is read."""

from __future__ import annotations

import numpy as np
import torch

from .model import GRID

LAYER = 8
NEUTRAL = "a photo of something"


def square_crop(point, W, H, side):
    """Axis-aligned square of the given side (clamped to the image) centred on the point and pushed inside the image."""
    w, h = min(side, W), min(side, H)
    x0 = min(max(point[0] - w / 2, 0), W - w)
    y0 = min(max(point[1] - h / 2, 0), H - h)
    return [int(round(x0)), int(round(y0)), int(round(x0 + w)), int(round(y0 + h))]


def grid_geometry(width: int, height: int, preprocess: str) -> tuple[int, int, int]:
    """(x offset, y offset, side) of the square the 24 x 24 grid covers: image pixel x of square column u = u - x offset.
    crop: the central min(W, H) square (negative offsets); pad: the padded square around the image."""
    if preprocess == "crop":
        s = min(width, height)
        return -((width - s) // 2), -((height - s) // 2), s
    if width == height:
        return 0, 0, width
    if width > height:
        return 0, (width - height) // 2, width
    return (height - width) // 2, 0, height


def image_cells(width: int, height: int, preprocess: str) -> np.ndarray:
    """(GRID, GRID) mask of the grid cells that show the image (with padding, the cells on padding only are False)."""
    if preprocess == "crop":
        return np.ones((GRID, GRID), bool)
    dx, dy, side = grid_geometry(width, height, preprocess)
    edges = np.linspace(0, side, GRID + 1)
    sq = np.clip(np.searchsorted(edges, np.arange(side) + 0.5) - 1, 0, GRID - 1)
    rows, cols = sq[dy:dy + height], sq[dx:dx + width]
    mask = np.zeros((GRID, GRID), bool)
    mask[np.ix_(np.unique(rows), np.unique(cols))] = True
    return mask


class SimilarityLocator:
    """Peak of the similarity map: per-image centred layer-8 image-token states against the centred text vector of
    "a photo of a {phrase}" minus "a photo of something"."""

    def __init__(self, runner):
        self.runner = runner
        self.cache: dict[str, torch.Tensor] = {}
        self.neutral = self.text_vec(NEUTRAL)

    @torch.no_grad()
    def text_vec(self, phrase: str) -> torch.Tensor:
        if phrase not in self.cache:
            hs = self.runner.forward(self.runner.tok(phrase, return_tensors="pt").input_ids).hidden_states
            self.cache[phrase] = hs[LAYER][0, -1].float()
        return self.cache[phrase]

    def peak(self, hidden_states, start: int, stop: int, phrase: str, W: int, H: int):
        """(x, y) in image pixels of the peak cell (image cells only) and the peak value."""
        h = hidden_states[LAYER][0, start:stop].float()
        h = torch.nn.functional.normalize(h - h.mean(0), dim=-1)
        v = torch.nn.functional.normalize(self.text_vec(phrase) - self.neutral, dim=-1)
        sim = (h @ v).cpu().numpy()
        mask = image_cells(W, H, self.runner.preprocess).reshape(-1)
        sim_in = np.where(mask, sim, -np.inf)
        i = int(np.argmax(sim_in))
        r, c = divmod(i, GRID)
        dx, dy, side = grid_geometry(W, H, self.runner.preprocess)
        px = min(max((c + 0.5) * side / GRID - dx, 0.0), float(W))
        py = min(max((r + 0.5) * side / GRID - dy, 0.0), float(H))
        return (px, py), float(sim_in.max())
