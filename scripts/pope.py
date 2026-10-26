"""P1 answers to POPE questions. Every (image, object) pair in the given question files is verified once (the splits share
images and positive objects), then each question gets that verdict. Resumable per image; shard over GPUs with --shard /
--num-shards.

    python scripts/pope.py --questions coco_pope_random.json coco_pope_popular.json coco_pope_adversarial.json \\
        --image-dir /path/to/coco/val2014 --out outputs/pope_coco_p1.jsonl
    python scripts/eval_pope.py outputs/pope_coco_p1.jsonl

The question files are POPE's own (github.com/RUCAIBox/POPE, output/coco and output/seem): JSON lines or one JSON array of
{question_id, image, text, label}. P1 asks question(object) for the object named in the text, with LLaVA-1.5's POPE
instruction "Answer the question using a single word or phrase."
Rows: {file, question_id, image, text, label, answer, greedy, s, m, c, m2}. The sampling seed is the image id (the digits at
the end of the file name).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1 import P1  # noqa: E402

OBJECT_RE = re.compile(r"(?:Is there an?|Are there) (.+?) in the im", re.IGNORECASE)


def load_questions(path: str) -> list[dict]:
    text = Path(path).read_text()
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", nargs="+", required=True, help="POPE question files")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--vision-tower", default=None)
    ap.add_argument("--preprocess", default="crop", choices=["crop", "pad"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    by_image: dict[str, list[tuple[str, dict, str]]] = {}
    for path in args.questions:
        for q in load_questions(path):
            obj = OBJECT_RE.match(q["text"]).group(1).lower()
            by_image.setdefault(q["image"], []).append((Path(path).name, q, obj))
    images = sorted(by_image, key=lambda name: int(re.search(r"(\d+)$", Path(name).stem).group(1)))
    images = [im for n, im in enumerate(images) if n % args.num_shards == args.shard]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["image"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [im for im in images if im not in done]
    print(f"[shard {args.shard}] {len(todo)} images", flush=True)

    p1 = P1.from_pretrained(args.model, vision_tower=args.vision_tower, preprocess=args.preprocess)
    with open(out, "a", encoding="utf-8") as f:
        for n, name in enumerate(todo, start=1):
            image = Image.open(Path(args.image_dir) / name).convert("RGB")
            seed = int(re.search(r"(\d+)$", Path(name).stem).group(1))
            objects = sorted({obj for _, _, obj in by_image[name]})
            verdicts = p1.answer(image, objects, seed=seed)
            for fname, q, obj in by_image[name]:
                v = verdicts[obj]
                f.write(json.dumps({"file": fname, "question_id": q.get("question_id"), "image": name, "text": q["text"],
                                    "label": q.get("label"), "answer": v["answer"], "greedy": v["greedy"],
                                    "s": round(v["s"], 4), "m": round(v["m"], 4), "c": v["c"],
                                    "m2": None if v["m2"] is None else round(v["m2"], 4)}) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"[shard {args.shard}] {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
