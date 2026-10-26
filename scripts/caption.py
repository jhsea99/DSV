"""P1 captions for a list of images, one JSON row per image (resumable; shard over GPUs with --shard / --num-shards).

    # your own images
    python scripts/caption.py --images photo1.jpg photo2.jpg --out outputs/captions.jsonl

    # CHAIR on OPERA's 500 COCO val2014 images (the paper setting)
    python scripts/caption.py --image-dir /path/to/coco/val2014 --image-ids data/chair_500_image_ids.txt \\
        --out outputs/chair_p1.jsonl

Rows: {image_id, image, caption, caption_withheld, samples, sentences, verdicts, stats, n_tokens, seconds}.
The sampling seed is the image id (the digits at the end of the file name, as in COCO), otherwise --seed.
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

COCO_NAME = "COCO_val2014_{:012d}.jpg"


def image_seed(path: Path, default: int) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", help="image files")
    src.add_argument("--image-ids", help="text file of COCO image ids, one per line (with --image-dir)")
    ap.add_argument("--image-dir", help="folder of the images named in --image-ids (COCO_val2014_<id>.jpg)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--vision-tower", default=None, help="local copy of openai/clip-vit-large-patch14-336 (optional)")
    ap.add_argument("--preprocess", default="crop", choices=["crop", "pad"])
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for files whose name does not end with digits")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if args.image_ids:
        if not args.image_dir:
            ap.error("--image-ids needs --image-dir")
        ids = sorted(int(x) for x in Path(args.image_ids).read_text().split())
        paths = [Path(args.image_dir) / COCO_NAME.format(i) for i in ids]
    else:
        paths = sorted(Path(p) for p in args.images)
    paths = [p for n, p in enumerate(paths) if n % args.num_shards == args.shard]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["image"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [p for p in paths if p.name not in done]
    print(f"[shard {args.shard}] {len(todo)} images to caption", flush=True)

    p1 = P1.from_pretrained(args.model, vision_tower=args.vision_tower, preprocess=args.preprocess)
    with open(out, "a", encoding="utf-8") as f:
        for n, path in enumerate(todo, start=1):
            image = Image.open(path).convert("RGB")
            seed = image_seed(path, args.seed)
            res = p1.caption(image, seed=seed)
            row = {"image_id": seed, "image": path.name, **res}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[shard {args.shard}] {n}/{len(todo)} {path.name} {res['seconds']}s rewrites {res['stats']['rewrites']}",
                  flush=True)


if __name__ == "__main__":
    main()
