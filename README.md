# Mitigating Object Hallucination in Vision-Language Models via Context-Detached Self-Verification

**Ji-Won Kim, Hae-Won Hwang, Byung-Cheol Song**, Department of Electrical and Computer Engineering, Inha University

Official code of **DSV (Context-Detached Self-Verification)**.

Vision-language models often mention objects that are not in the image when they write detailed captions. Yet the same
model judges object existence fairly accurately when it is asked in a new context, detached from the caption. DSV uses
that judgment during caption generation, with a single VLM and no additional training, labels or external models.
After each sentence, DSV judges the mentioned objects by combining a context-detached existence question with the
consensus of sampled captions, and rewrites unsupported ones with existence-weighted decoding.

On LLaVA-1.5-7B, DSV reduces CHAIR<sub>s</sub> from 47.6% to 30.8% and CHAIR<sub>i</sub> from 13.2% to 7.4% while
preserving caption length, and achieves the highest accuracy in 8 of 9 POPE settings.

## Method

**Sentence-level verification loop.** DSV first samples K = 4 captions per image at temperature T = 1.2. The main caption
is generated greedily, one sentence at a time. At the end of each sentence, the same model lists, from the sentence alone,
the physical objects it mentions (no word list, no detector). An object whose score is negative is rejected: the singular
and plural forms of its word are banned until the end of the caption, and the sentence is rewound to its start and
regenerated. A rewritten sentence is verified the same way, at most three times.

**Combining two pieces of evidence.** For an object X:

- the context-detached question asks *"Is there a X in the image?"* with the image in a new context without the caption
  (with LLaVA-1.5's short-answer instruction) and takes the log-odds of Yes over No at the first answer token, `m`;
- the sample consensus counts the sampled captions that mention X, `c`, as the log-odds `l = log((c + 1) / (K - c + 1))`;
- when `m + l < 0`, the same question is asked again on a crop around X, `m2`. X is located where the image-token
  representations of an intermediate layer are most similar to the text representation of "a photo of a X".

```
s = max(m, m2) + l    (m2 only when the crop question was asked; otherwise s = m + l)
```

X is rejected when `s < 0`, that is, when the question's odds times the samples' odds is below 1. On POPE, whose questions
have the same form as the context-detached question, DSV answers Yes when `s >= 0`.

**Existence-weighted rewriting.** While a rejected sentence is rewritten, the logit of every candidate token that leads to
a noun gets `log sigmoid(s)` of that noun added. Whether a candidate leads to a noun is decided by generating up to two
words greedily from it and POS-tagging them. The weighting is applied only while rewriting, so the original generation
path is unchanged.

## Results (LLaVA-1.5-7B)

All methods use the same preprocessing (shortest side resized to 336, centre crop). VCD, OPERA, PAI and SumGD were run
with their official code and paper settings.

**Table 1.** CHAIR on OPERA's 500 COCO val2014 images (%), "Please describe this image in detail.", at most 512 tokens.

| Method | CHAIR<sub>s</sub> | CHAIR<sub>i</sub> | Recall | Words |
|---|---|---|---|---|
| Greedy | 47.6 | 13.2 | 80.5 | 100 |
| VCD | 55.8 | 16.0 | 78.7 | 102 |
| OPERA | 47.0 | 13.2 | 79.7 | 92 |
| PAI | 43.0 | 11.3 | **81.2** | 97 |
| SumGD | 40.8 | 10.1 | 79.6 | 101 |
| **DSV (Ours)** | **30.8** | **7.4** | 78.8 | 100 |

**Table 2.** POPE accuracy (%), 500 images and 3,000 questions per setting.

| Dataset | Setting | Greedy | VCD | OPERA | PAI | DSV |
|---|---|---|---|---|---|---|
| COCO | Random | 87.0 | 85.4 | 86.5 | 86.5 | **89.4** |
| | Popular | 85.9 | 83.3 | 85.4 | 85.4 | **87.2** |
| | Adversarial | 83.6 | 80.6 | 83.3 | 83.3 | **85.4** |
| A-OKVQA | Random | 88.7 | 85.8 | 88.2 | 88.2 | **89.6** |
| | Popular | 85.4 | 80.5 | 85.1 | 85.2 | **87.2** |
| | Adversarial | 79.2 | 74.4 | 79.2 | 79.3 | **82.2** |
| GQA | Random | **89.5** | 85.1 | 88.2 | 88.4 | 88.4 |
| | Popular | 84.3 | 78.1 | 83.0 | 83.2 | **86.0** |
| | Adversarial | 81.0 | 74.7 | 80.7 | 80.8 | **82.5** |

## Installation

Python 3.10 or 3.11 (tested with 3.11) and a CUDA GPU with 24 GB of memory (tested on an RTX 4090; peak use is about
23 GB).

```bash
git clone https://github.com/jhsea99/DSV.git && cd DSV
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
```

`pip install -e .` installs the official LLaVA code (github.com/haotian-liu/LLaVA, pinned commit), which pins
`torch==2.1.2` and `transformers==4.37.2`. The model weights (`liuhaotian/llava-v1.5-7b`, about 14 GB) and the vision
tower (`openai/clip-vit-large-patch14-336`) are downloaded from the Hugging Face Hub on first use.

## Quick start

```python
from PIL import Image
from dsv import DSV

dsv = DSV.from_pretrained("liuhaotian/llava-v1.5-7b")
image = Image.open("example.jpg").convert("RGB")

out = dsv.caption(image)
print(out["caption"])

print(dsv.answer(image, ["dog", "frisbee"]))
# {'dog': {'answer': 'yes', 's': ..., ...}, 'frisbee': {'answer': 'no', ...}}
```

From the command line:

```bash
python scripts/caption.py --images example1.jpg example2.jpg --out outputs/captions.jsonl
```

Each output row has `caption`, `samples`, `sentences` (every version of every sentence with its objects, scores and
rejections), `verdicts` (per object: `m`, `c`, `m2`, `s` and the crop rectangle) and `stats`. It also has
`caption_withheld`, an optional output not used in the paper: the caption without the sentences that were still rejected
after the third rewrite.

| Option | Default | Meaning |
|---|---|---|
| `model_path` / `--model` | `liuhaotian/llava-v1.5-7b` | LLaVA-1.5 checkpoint (Hub id or local folder) |
| `vision_tower` / `--vision-tower` | from the checkpoint | local copy of `openai/clip-vit-large-patch14-336` |
| `preprocess` / `--preprocess` | `crop` | `crop`: shortest side 336 and centre crop (the paper). `pad`: pad to a square first (LLaVA-1.5's own setting) |
| `seed` | image id or 0 | seed of the sampled captions |

## Reproducing the paper

### CHAIR (Table 1)

Data: COCO val2014 images and the COCO 2014 annotations (`captions_{train,val}2014.json`,
`instances_{train,val}2014.json`). `data/chair_500_image_ids.txt` lists the 500 images of OPERA's CHAIR evaluation
(github.com/shikiw/OPERA, `log/chair_eval_results/llava-1.5/`).

```bash
# one process per GPU; --shard i --num-shards N splits the images
python scripts/caption.py --image-dir /path/to/coco/val2014 --image-ids data/chair_500_image_ids.txt \
    --out outputs/chair_dsv_0.jsonl --shard 0 --num-shards 2
python scripts/caption.py --image-dir /path/to/coco/val2014 --image-ids data/chair_500_image_ids.txt \
    --out outputs/chair_dsv_1.jsonl --shard 1 --num-shards 2
cat outputs/chair_dsv_[01].jsonl > outputs/chair_dsv.jsonl
```

Score with the CHAIR script of OPERA (`chair.py` in github.com/shikiw/OPERA):

```bash
python chair.py --cap_file outputs/chair_dsv.jsonl --caption_key caption --coco_path /path/to/coco/annotations
```

### POPE (Table 2)

Question files: github.com/RUCAIBox/POPE, `output/coco/coco_pope_{random,popular,adversarial}.json` (COCO val2014
images) and `output/seem/{aokvqa,gqa}/*_pope_seem_{random,popular,adversarial}.json` (A-OKVQA: COCO val2014 images; GQA:
GQA images).

```bash
python scripts/pope.py --questions coco_pope_random.json coco_pope_popular.json coco_pope_adversarial.json \
    --image-dir /path/to/coco/val2014 --out outputs/pope_coco_dsv.jsonl
python scripts/eval_pope.py outputs/pope_coco_dsv.jsonl
```

`eval_pope.py` prints accuracy, precision, recall and F1 per question file for DSV and for plain LLaVA (`greedy`, the
answer on the same logits). Pass all settings of a dataset together: each (image, object) pair is verified once.

### Determinism and cost

Decoding is deterministic (greedy, fp16, eager attention) and the samples are seeded with the image id, so the same GPU
type and library versions give the same captions. On an RTX 4090, a caption takes about 13.4 s (4.4 s for the samples)
and a POPE image with about ten questions about 6.5 s.

## Repository layout

```
dsv/
  __init__.py   DSV: caption(), answer(), samples()
  model.py      LLaVA-1.5 loading, prompts, preprocessing, yes/no log-odds
  verify.py     the DSV verdict (context-detached question, sample consensus, crop question)
  locate.py     similarity peak and square crop
  decode.py     sentence-level verification loop and existence-weighted rewriting
  samples.py    the sampled captions
  text.py       prompts and text helpers
scripts/
  caption.py    captions for image files or a COCO id list
  pope.py       POPE answers
  eval_pope.py  POPE metrics
data/
  chair_500_image_ids.txt
```

## Citation

```bibtex
@inproceedings{kim2026dsv,
  title     = {Mitigating Object Hallucination in Vision-Language Models via Context-Detached Self-Verification},
  author    = {Kim, Ji-Won and Hwang, Hae-Won and Song, Byung-Cheol},
  booktitle = {Proceedings of the Autumn Annual Conference of IEIE},
  year      = {2026}
}
```

Korean title: 생성 문맥에서 분리된 자기 검증을 통한 시각-언어 모델의 객체 환각 완화, 대한전자공학회 2026년도 추계학술대회 ([conf.theieie.org/2026f](https://conf.theieie.org/2026f/)).

## License

Copyright 2026 Ji-Won Kim, Hae-Won Hwang, Byung-Cheol Song. Released under the [Apache License 2.0](LICENSE), the license of LLaVA, on which this code builds.

## Acknowledgements

This code builds on [LLaVA](https://github.com/haotian-liu/LLaVA). The evaluation follows
[POPE](https://github.com/RUCAIBox/POPE) and the CHAIR protocol of [OPERA](https://github.com/shikiw/OPERA).
