"""P1 caption decoding.

The caption is written greedily, one sentence at a time. At each sentence end:
  1. the model lists, text only, the physical objects the sentence mentions (a listed phrase is kept only when its head
     word occurs in the sentence);
  2. each phrase gets the P1 verdict (p1/verify.py);
  3. if a phrase is rejected, its words are banned for the rest of the caption, the sentence is rewound to its start and
     rewritten. While rewriting, every candidate token that begins a noun (nltk NN / NNS under a two-word greedy lookahead)
     gets log sigmoid(s) of that noun added to its logit, so the rewrite prefers objects the verdict supports;
  4. a sentence is rewritten at most three times; a rewrite that repeats an earlier version is accepted as is.
"caption" is the decoded text. "caption_withheld" leaves out the sentences that were still rejected after rewriting (the
model's context is unchanged: it continues as if it had said them).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from PIL import Image

from .text import CAPTION_PROMPT, EXTRACT_PROMPT, SENT_END, SKIP_LOOKAHEAD, WORD_RE, forms, head, parse_phrases, surface
from .verify import ImageVerifier


class CaptionDecoder:
    def __init__(self, runner, locator, max_new_tokens: int = 512, max_rewrites: int = 3):
        import nltk  # noqa: F401  (fails early when nltk is missing)

        self.r, self.loc = runner, locator
        self.max_new_tokens, self.max_rewrites = max_new_tokens, max_rewrites
        tok = runner.tok
        self.prompt = runner.prompt_ids(CAPTION_PROMPT, suffix="").to(runner.device)
        self.eos = tok.eos_token_id
        self.pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
        self.alpha_start = [p is not None and p.startswith("▁") and p[1:].isalpha() for p in self.pieces]
        self.cont = [p is not None and p.isalpha() for p in self.pieces]
        self.sent_end = [p is not None and p.rstrip().endswith(SENT_END) for p in self.pieces]

    # ---- model steps ----
    @torch.no_grad()
    def step(self, token: int, pkv):
        o = self.r.model(input_ids=torch.tensor([[token]], device=self.r.device), past_key_values=pkv, use_cache=True)
        return o.logits[0, -1].float(), o.past_key_values

    @staticmethod
    def truncate(pkv, n: int):
        return tuple((k[:, :, :n, :], v[:, :, :n, :]) for k, v in pkv)

    @torch.no_grad()
    def extract(self, sentence: str) -> tuple[str, list[str]]:
        tok, model = self.r.tok, self.r.model
        ids = self.r.prompt_ids(EXTRACT_PROMPT.format(s=sentence), with_image=False).to(self.r.device)
        gen = model.generate(inputs=ids, max_new_tokens=40, use_cache=True, do_sample=False, num_beams=1)
        text = tok.batch_decode(gen[:, ids.shape[1]:] if gen.shape[1] > ids.shape[1] and gen[0, 0] == ids[0, 0] else gen,
                                skip_special_tokens=True)[0].strip()
        return text, parse_phrases(text)

    @torch.no_grad()
    def complete_word(self, prefix: list[int], pkv) -> str:
        """The word the pieces `prefix` (all but the last already in the cache) start under greedy continuation (up to 4
        continuation pieces); the cache is not advanced."""
        toks = list(prefix)
        z, p = self.step(prefix[-1], pkv)
        for _ in range(4):
            nx = int(z.argmax())
            if not self.cont[nx]:
                break
            toks.append(nx)
            z, p = self.step(nx, p)
        return self.r.tok.decode(toks, skip_special_tokens=True).strip().lower()

    def word_in_progress(self, gen: list[int]) -> list[int]:
        partial: list[int] = []
        for tk in reversed(gen):
            if self.alpha_start[tk]:
                partial.insert(0, tk)
                break
            if not self.cont[tk]:
                break
            partial.insert(0, tk)
        return partial if partial and self.alpha_start[partial[0]] else []

    # ---- one caption ----
    def __call__(self, image: Image.Image, sample_sets: list[set[str]]) -> dict:
        import nltk

        t0 = time.time()
        r, tok, model = self.r, self.r.tok, self.r.model
        alpha_start, cont, pieces, eos = self.alpha_start, self.cont, self.pieces, self.eos
        ver = ImageVerifier(r, self.loc, image, sample_sets)
        banned: dict[str, str] = {}  # surface form -> rejected phrase
        stats = {"rewrites": 0, "cap_hits": 0, "cycles": 0, "withheld": 0}

        def pick(z, pkv, gen):
            """Greedy token with the bans: a token that starts, or continues, a word whose greedy completion is a banned
            form is banned, then the argmax is taken again ("cell" + "phone" cannot evade a ban on "phone")."""
            if not banned:
                return int(z.argmax())
            z = z.clone()
            seen: set[int] = set()
            partial = self.word_in_progress(gen)
            for _ in range(32):
                t = int(z.argmax())
                if t == eos or t in seen:
                    return t
                if alpha_start[t]:
                    prefix = [t]
                elif cont[t] and partial:
                    prefix = partial + [t]
                else:
                    return t
                first = tok.decode(prefix, skip_special_tokens=True).strip().lower()
                if not any(fm.startswith(first) for fm in banned):
                    return t
                if self.complete_word(prefix, pkv) not in banned:
                    return t
                seen.add(t)
                z[t] = -math.inf
            return int(z.argmax())

        def banned_token(t, partial, pkv):
            if not banned:
                return False
            if alpha_start[t]:
                prefix = [t]
            elif cont[t] and partial:
                prefix = partial + [t]
            else:
                return False
            first = tok.decode(prefix, skip_special_tokens=True).strip().lower()
            if not any(fm.startswith(first) for fm in banned):
                return False
            return self.complete_word(prefix, pkv) in banned

        @torch.no_grad()
        def noun_run_ahead(t, pkv, gen, s0):
            """The POS noun run (NN / NNS, at most two words) that candidate t begins under a greedy two-word lookahead."""
            look, words = [t], 1
            z_, p_ = self.step(t, pkv)
            for _ in range(8):
                nx = int(z_.argmax())
                if nx == eos or not (alpha_start[nx] or cont[nx]):
                    break
                if alpha_start[nx]:
                    words += 1
                    if words > 2:
                        break
                look.append(nx)
                z_, p_ = self.step(nx, p_)
            before = tok.decode(gen[s0:], skip_special_tokens=True)
            ahead = tok.decode(look, skip_special_tokens=True).strip()
            toks_b = nltk.word_tokenize(before)
            tagged = nltk.pos_tag(toks_b + nltk.word_tokenize(ahead))
            run = []
            for w_, tg in tagged[len(toks_b):len(toks_b) + 2]:
                if tg in ("NN", "NNS") and w_.isalpha():
                    run.append(w_.lower())
                else:
                    break
            return " ".join(run) if run else None

        def pick_mult(z, pkv, gen, s0):
            """argmax of logit + log sigmoid(s(noun run)) over candidates in logit order (exact: the additions are <= 0)."""
            partial = self.word_in_progress(gen)
            vals, order = torch.sort(z, descending=True, stable=True)
            best, best_score = None, -math.inf
            for val, t in zip(vals[:64].tolist(), order[:64].tolist()):
                if val <= best_score:
                    break
                adj = 0.0
                if t != eos:
                    if banned_token(t, partial, pkv):
                        continue
                    if alpha_start[t] and pieces[t][1:].lower() not in SKIP_LOOKAHEAD:
                        run = noun_run_ahead(t, pkv, gen, s0)
                        if run:
                            adj = float(F.logsigmoid(torch.tensor(ver.verdict(run)["s"])))
                if val + adj > best_score:
                    best, best_score = t, val + adj
            return pick(z, pkv, gen) if best is None else best

        with torch.no_grad():
            pix = r.pixel_values(image).unsqueeze(0).to(r.device, torch.float16)
            o = model(input_ids=self.prompt, images=pix, image_sizes=[image.size], use_cache=True)
        pkv, base = o.past_key_values, o.past_key_values[0][0].shape[2]
        hist = [o.logits[0, -1].float()]
        gen: list[int] = []
        sentences, versions = [], []
        sent_start, rewrites_here = 0, 0
        while len(gen) < self.max_new_tokens:
            t = pick_mult(hist[len(gen)], pkv, gen, sent_start) if rewrites_here > 0 else pick(hist[len(gen)], pkv, gen)
            if t != eos:
                gen.append(t)
                z, pkv = self.step(t, pkv)
                hist.append(z)
                if not self.sent_end[t]:
                    continue
            text = tok.decode(gen[sent_start:], skip_special_tokens=True).strip()
            if text:
                raw_list, phrases = self.extract(text)
                words = {w.lower() for w in WORD_RE.findall(text)}
                kept = [p for p in phrases if surface(p, words)]
                vs = {p: ver.verdict(p) for p in kept}
                rejected = [p for p in kept if vs[p]["rejected"]]
                version = {"text": text, "objects": kept, "s": {p: round(v["s"], 3) for p, v in vs.items()},
                           "rejected": rejected}
                cycle = text in {v["text"] for v in versions}
                if rejected and rewrites_here < self.max_rewrites and not cycle:
                    for p in rejected:
                        for w in surface(p, words) | {head(p)}:
                            for fm in forms(w):
                                banned.setdefault(fm, p)
                    stats["rewrites"] += 1
                    rewrites_here += 1
                    versions.append(version)
                    gen = gen[:sent_start]
                    hist = hist[: sent_start + 1]
                    pkv = self.truncate(pkv, base + sent_start)
                    continue
                if rejected:
                    stats["cap_hits" if rewrites_here >= self.max_rewrites else "cycles"] += 1
                versions.append(version)
                withheld = bool(rejected) and len(versions) > 1
                stats["withheld"] += withheld
                sentences.append({"text": text, "versions": versions, "withheld": withheld})
            if t == eos:
                break
            sent_start, rewrites_here, versions = len(gen), 0, []
        caption = tok.decode(gen, skip_special_tokens=True).strip()
        stats.update(ver.stats)
        return {
            "caption": caption,
            "caption_withheld": " ".join(s["text"] for s in sentences if not s["withheld"]),
            "sentences": sentences,
            "verdicts": {p: {"m": round(v["m"], 3), "c": v["c"], "m2": None if v["m2"] is None else round(v["m2"], 3),
                             "s": round(v["s"], 3), "rect": v["rect"]} for p, v in ver.verdicts.items()},
            "stats": stats,
            "n_tokens": len(gen),
            "seconds": round(time.time() - t0, 2),
        }
