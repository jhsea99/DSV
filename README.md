# P1: Verified Rollback Decoding for LLaVA-1.5

P1 reduces object hallucination in image captions at decoding time. It needs no training, no extra model and no external
detector: LLaVA-1.5 checks its own sentences and rewrites the ones that mention objects it cannot confirm.

On CHAIR (OPERA's 500 COCO images) P1 lowers CHAIR<sub>s</sub> from 47.6 to 30.8 and CHAIR<sub>i</sub> from 13.2 to 7.4
at the same caption length. On POPE it raises accuracy on COCO, A-OKVQA and GQA.

## How it works

The caption is written greedily, one sentence at a time. At the end of each sentence:

1. **List.** The model lists, text only, the physical objects the sentence mentions.
2. **Verify.** Each object `p` gets a score

   ```
   s(p) = q(p) + log((c + 1) / (K - c + 1))
   ```

   - `m` is the log-odds of "Yes" over "No" for *"Is there a p in the image?"*, asked in a fresh context.
   - `c` is how many of `K = 4` captions sampled at temperature 1.2 mention `p`.
   - `q(p) = m`, unless `m + log-odds < 0`. Then the question is asked again on a crop around the image region most
     similar to `p` (layer-8 hidden states, no attention maps, no generated coordinates) and `q(p) = max(m, m2)`.

   The object is accepted when `s(p) >= 0`. In odds, this multiplies the question's odds by the samples' odds.
3. **Rewrite.** If an object is rejected, its word is banned for the rest of the caption and the sentence is rewound and
   rewritten. While rewriting, every candidate token that starts a noun gets `log sigmoid(s(noun))` added to its logit,
   so the rewrite prefers objects the verifier supports.
4. A sentence is rewritten at most three times. **Withholding** (optional output) drops the sentences that are still
   rejected after that. The model's context is unchanged.

For POPE, P1's answer is the verdict alone: "yes" when `s(object) >= 0`.

## Results (LLaVA-1.5-7B)

All methods use the same prompts and the preprocessing of the baselines' evaluation code (shortest side resized to 336,
centre crop). Baselines were run with their official code and settings.

**CHAIR**, OPERA's 500 COCO val2014 images, "Please describe this image in detail.", at most 512 new tokens.
Lower CHAIR is better.

| Method | CHAIR<sub>s</sub> | CHAIR<sub>i</sub> | Recall | Words |
|---|---|---|---|---|
| Greedy | 47.6 | 13.2 | 80.5 | 100 |
| VCD | 55.8 | 16.0 | 78.7 | 102 |
| OPERA | 47.0 | 13.2 | 79.7 | 92 |
| PAI | 43.0 | 11.3 | 81.2 | 97 |
| SumGD | 40.8 | 10.1 | 79.6 | 101 |
| **P1** | **30.8** | **7.4** | 78.8 | 100 |
| **P1 + withholding** | **28.8** | **6.9** | 78.3 | 94 |

**POPE accuracy**, 500 images and 3,000 questions per split, POPE labels.

| Method | COCO rand. | COCO pop. | COCO adv. | A-OKVQA rand. | A-OKVQA pop. | A-OKVQA adv. | GQA rand. | GQA pop. | GQA adv. |
|---|---|---|---|---|---|---|---|---|---|
| Greedy | 87.0 | 85.9 | 83.6 | 88.7 | 85.4 | 79.2 | **89.5** | 84.3 | 81.0 |
| VCD | 85.4 | 83.3 | 80.6 | 85.8 | 80.5 | 74.4 | 85.1 | 78.1 | 74.7 |
| OPERA | 86.5 | 85.4 | 83.3 | 88.2 | 85.1 | 79.2 | 88.2 | 83.0 | 80.7 |
| PAI | 86.5 | 85.4 | 83.3 | 88.2 | 85.2 | 79.3 | 88.4 | 83.2 | 80.8 |
| **P1** | **89.4** | **87.2** | **85.4** | **89.6** | **87.2** | **82.2** | 88.4 | **86.0** | **82.5** |

## Installation

Python 3.10 or 3.11 (tested with 3.11) and a CUDA GPU with 24 GB of memory (tested on an RTX 4090; peak use is about 23 GB).

```bash
git clone <this repository> && cd <this repository>
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
from p1 import P1

p1 = P1.from_pretrained("liuhaotian/llava-v1.5-7b")
image = Image.open("example.jpg").convert("RGB")

out = p1.caption(image)
print(out["caption"])            # P1 caption
print(out["caption_withheld"])   # P1 + withholding

print(p1.answer(image, ["dog", "frisbee"]))
# {'dog': {'answer': 'yes', 's': ..., ...}, 'frisbee': {'answer': 'no', ...}}
```

From the command line:

```bash
python scripts/caption.py --images example1.jpg example2.jpg --out outputs/captions.jsonl
```

Each output row has `caption`, `caption_withheld`, `samples`, `sentences` (every version of every sentence with its
objects, scores and rejections), `verdicts` (per object: `m`, `c`, `m2`, `s` and the crop rectangle) and `stats`.

Options of `P1.from_pretrained` and the scripts:

| Option | Default | Meaning |
|---|---|---|
| `model_path` / `--model` | `liuhaotian/llava-v1.5-7b` | LLaVA-1.5 checkpoint (Hub id or local folder) |
| `vision_tower` / `--vision-tower` | from the checkpoint | local copy of `openai/clip-vit-large-patch14-336` |
| `preprocess` / `--preprocess` | `crop` | `crop`: shortest side 336 and centre crop (the tables above). `pad`: pad to a square first (LLaVA-1.5's own setting) |
| `seed` | image id or 0 | seed of the four sampled captions |

## Reproducing the paper numbers

### CHAIR

Data: COCO val2014 images and the COCO 2014 annotations (`captions_{train,val}2014.json`,
`instances_{train,val}2014.json`). The image list `data/chair_500_image_ids.txt` is the 500 images of OPERA's CHAIR
evaluation (github.com/shikiw/OPERA, `log/chair_eval_results/llava-1.5/`).

```bash
# one process per GPU; --shard i --num-shards N splits the images
python scripts/caption.py --image-dir /path/to/coco/val2014 --image-ids data/chair_500_image_ids.txt \
    --out outputs/chair_p1_0.jsonl --shard 0 --num-shards 2
python scripts/caption.py --image-dir /path/to/coco/val2014 --image-ids data/chair_500_image_ids.txt \
    --out outputs/chair_p1_1.jsonl --shard 1 --num-shards 2
cat outputs/chair_p1_[01].jsonl > outputs/chair_p1.jsonl
```

Score with the CHAIR script of OPERA (`chair.py` in github.com/shikiw/OPERA):

```bash
python chair.py --cap_file outputs/chair_p1.jsonl --caption_key caption --coco_path /path/to/coco/annotations
python chair.py --cap_file outputs/chair_p1.jsonl --caption_key caption_withheld --coco_path /path/to/coco/annotations
```

### POPE

Question files: github.com/RUCAIBox/POPE, `output/coco/coco_pope_{random,popular,adversarial}.json` (COCO val2014
images) and `output/seem/{aokvqa,gqa}/*_pope_seem_{random,popular,adversarial}.json` (A-OKVQA: COCO val2014 images; GQA:
GQA images).

```bash
python scripts/pope.py --questions coco_pope_random.json coco_pope_popular.json coco_pope_adversarial.json \
    --image-dir /path/to/coco/val2014 --out outputs/pope_coco_p1.jsonl
python scripts/eval_pope.py outputs/pope_coco_p1.jsonl
```

`eval_pope.py` prints accuracy, precision, recall and F1 per question file for P1 and for plain LLaVA (`greedy`, the
answer on the same logits). All splits of a dataset should be passed together: each (image, object) pair is verified once.

### Determinism and cost

Decoding is deterministic (greedy, fp16, eager attention) and the samples are seeded with the image id, so the same GPU
type and library versions give the same captions. On an RTX 4090, a caption takes about 13 s (4.4 s for the samples and
9 s for decoding and verification) and a POPE image with ten questions about 6.5 s.

## Repository layout

```
p1/
  __init__.py   P1: caption(), answer(), samples()
  model.py      LLaVA-1.5 loading, prompts, preprocessing, yes/no margin
  verify.py     the P1 verdict (question x sample consensus, crop second look)
  locate.py     similarity peak and square crop
  decode.py     sentence-level rollback decoding with bans and noun multiplication
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

To be added.
