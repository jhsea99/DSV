"""Accuracy, precision, recall and F1 (yes = positive) of scripts/pope.py outputs, per question file, for DSV and for the
plain LLaVA answer on the same logits (greedy).

    python scripts/eval_pope.py outputs/pope_coco_dsv.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def scores(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    tp = sum(p and y for p, y in pairs)
    fp = sum(p and not y for p, y in pairs)
    fn = sum(not p and y for p, y in pairs)
    tn = sum(not p and not y for p, y in pairs)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / max(len(pairs), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "n": len(pairs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", nargs="+")
    args = ap.parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in args.outputs:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                groups[r["file"]].append(r)
    print(f"{'file':40s} {'method':7s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
    for name in sorted(groups):
        rows = groups[name]
        for method in ("greedy", "answer"):
            s = scores([(r[method] == "yes", r["label"] == "yes") for r in rows])
            label = "DSV" if method == "answer" else "greedy"
            print(f"{name:40s} {label:7s} {100 * s['accuracy']:6.2f} {100 * s['precision']:6.2f} {100 * s['recall']:6.2f} "
                  f"{100 * s['f1']:6.2f} {s['n']:6d}")


if __name__ == "__main__":
    main()
