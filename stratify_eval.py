"""Scores broken down by novelty of the test compound with respect to the training data."""
import argparse
import json
from pathlib import Path
from cache_predictions import load_predictions
from type_classifier import detect_and_load, load_vcd_dir, scheme, score_labels

CATS = ["1.1", "1.2", "1.2a", "1.2b", "2", "0"]
IAST = set("abcdefghijklmnopqrstuvwxyzāīūṛṝḷṅñṭḍṇśṣḥṃṁ")


def clean(s):
    return "".join(c for c in s if c in IAST)

def vcd_identity(r):
    return clean(r.get("lemma1", "")), clean(r.get("lemma2", ""))


def sacti_identity(r):
    cpd = r.get("compound", "")
    if "-" not in cpd:
        return clean(r.get("surface", "")), ""
    a, b = cpd.split("-", 1)
    return clean(a), clean(b)


def train_rubric(paths):
    """Members, compound strings and member pairs of the training data (train split only for a VCD directory)."""
    members, surfaces, pairs, n = set(), set(), set(), 0
    for path in paths:
        records = load_vcd_dir(path)[0] if Path(path).is_dir() else detect_and_load(path)
        ident = vcd_identity if any(r.get("lemma1") and r.get("lemma2") for r in records[:50]) else sacti_identity
        for r in records:
            k1, k2 = ident(r)
            members.update(k for k in (k1, k2) if k)
            surfaces.add(r.get("compound", ""))
            if k1 and k2:
                pairs.add((k1, k2))
        n += len(records)
    return members, surfaces, pairs, n


def categorize(eval_r, dataset, rubric):
    ident = vcd_identity if dataset == "vcd" else sacti_identity
    members, surfaces, pairs, _ = rubric
    cats = []
    for r in eval_r:
        k1, k2 = ident(r)
        seen = (bool(k1) and k1 in members) + (bool(k2) and k2 in members)
        if seen == 2:
            cats.append("1.1" if r.get("compound", "") in surfaces else "1.2a" if (k1, k2) in pairs else "1.2b")
        else:
            cats.append("2" if seen else "0")
    return cats


def strata(cats):
    return {c: [i for i, x in enumerate(cats) if x == c or (c == "1.2" and x in ("1.2a", "1.2b"))] for c in CATS}


def summarise(gold, pred, labels):
    if not gold:
        return {"n": 0}
    s = score_labels(gold, pred, labels)
    return {"n": len(gold), "acc": float(s["acc"]), "macro_f1": float(s["macro_f1"]),
            "macro_p": float(s["macro_p"]), "macro_r": float(s["macro_r"]),
            "per_class_f1": {l: {"p": round(float(s["p"][i]), 4), "r": round(float(s["r"][i]), 4),
                                 "f1": round(float(s["f1"][i]), 4), "support": int(s["support"][i])}
                             for i, l in enumerate(labels)},
            "cm": {g: {p: int(s["cm"][i, j]) for j, p in enumerate(labels) if s["cm"][i, j]}
                   for i, g in enumerate(labels) if s["support"][i]}}


def main(args):
    eval_r = detect_and_load(args.eval_data)
    per_record = load_predictions(args.predictions, eval_r, args.batch_size, args.use_lookup)
    labels, norm = scheme(args.merge_exo)
    gold = [norm(p["gold_type"]) for p in per_record]
    pred = [norm(p["pred_type"]) for p in per_record]
    rubric = train_rubric(args.train_data)
    groups = strata(categorize(eval_r, args.dataset, rubric))
    results = {c: summarise([gold[i] for i in ix], [pred[i] for i in ix], labels) for c, ix in groups.items()}
    overall = summarise(gold, pred, labels)

    print(f"{'category':8} {'n':>5}  {'acc':>6}  {'macro-F1':>8}")
    for c, r in [*results.items(), ("overall", overall)]:
        if r["n"]:
            print(f"{c:8} {r['n']:5d}  {100 * r['acc']:6.2f}  {100 * r['macro_f1']:8.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "dataset": args.dataset, "checkpoint": args.predictions, "train_paths": args.train_data,
        "eval_path": args.eval_data, "n_train": rubric[3], "n_eval": len(eval_r),
        "category_counts": {c: len(ix) for c, ix in groups.items()},
        "stratified_results": results,
        "overall_accuracy": overall["acc"], "overall_macro_f1": overall["macro_f1"],
        "overall_macro_p": overall["macro_p"], "overall_macro_r": overall["macro_r"],
        "primary_metric": "merged-exo" if args.merge_exo else "5-class",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["vcd", "sacti"])
    p.add_argument("--predictions", required=True, help="prediction cache (.json) or checkpoint")
    p.add_argument("--train-data", required=True, nargs="+", help="training sources that define 'seen'")
    p.add_argument("--eval-data", required=True)
    p.add_argument("--merge-exo", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--use-lookup", action="store_true")
    p.add_argument("--output", required=True)
    main(p.parse_args())
