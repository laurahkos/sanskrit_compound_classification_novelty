import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cache_predictions import load_predictions
from seed_compare import TAGS
from type_classifier import detect_and_load, score_labels

SETS = {
    "explicit_all": {"T3", "T4", "T7"},
    "explicit_noT3": {"T4", "T7"},
    "explicit_T3only": {"T3"},
    "kar_covert": {"U"},
    "nonkaraka": {"T6", "Tn", "Tp", "K1", "K6", "K7", "Km", "Bs6", "BvS", "Di"},
    "nonkaraka_tatpur": {"T6", "Tn", "Tp"},
}
GAPS = [("explicit_all", "nonkaraka"), ("explicit_noT3", "nonkaraka"), ("explicit_noT3", "nonkaraka_tatpur"),
        ("explicit_noT3", "kar_covert"), ("kar_covert", "nonkaraka")]


def metrics(ix, gold, pred):
    g, p = [gold[i] for i in ix], [pred[i] for i in ix]
    return {"n": len(ix), "acc": float(np.mean([a == b for a, b in zip(g, p)])) if ix else None,
            # single-class macro-F1 is inflated, so it needs at least two gold classes
            "macro_f1": float(score_labels(g, p)["macro_f1"]) if len(set(g)) > 1 else None,
            "gold_classes": dict(Counter(g))}


def gap_ci(ix_a, ix_b, ok, n_boot, seed=42):
    """acc(A) - acc(B) with an item bootstrap CI; the two sets are resampled independently."""
    if not ix_a or not ix_b:
        return {"diff": None, "ci": None}
    rng = np.random.default_rng(seed)
    a, b = ok[ix_a], ok[ix_b]
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return {"diff": float(a.mean() - b.mean()), "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}


def sources(args):
    """(key, src): key is TAG with --model, cond/seed/TAG with --cache-root."""
    if args.model:
        return [tuple(spec.split("=", 1)) for spec in args.model]
    return [(f"{cond}/{d.name}/{tag}", str(d / f"{tag}_sacti.json"))
            for cond in args.conds for d in sorted(Path(args.cache_root, cond).glob("seed*"))
            if not args.seeds or d.name in args.seeds
            for tag in TAGS if (d / f"{tag}_sacti.json").exists()]


def main(args):
    eval_r = detect_and_load(args.eval_data)
    lab = [r["_conll_label"] for r in eval_r]
    labels = sorted(set(lab))
    ix_of = {k: [i for i, l in enumerate(lab) if l in s] for k, s in SETS.items()}
    ix_label = {l: [i for i, x in enumerate(lab) if x == l] for l in labels}
    srcs = dict(sources(args))
    if not srcs:
        raise SystemExit("no predictions found")

    out = {}
    for key, src in srcs.items():
        pr = load_predictions(src, eval_r, args.batch_size)
        gold, pred = [r["gold_type"] for r in pr], [r["pred_type"] for r in pr]
        ok = np.array([g == p for g, p in zip(gold, pred)], float)
        out[key] = {**{k: metrics(ix, gold, pred) for k, ix in ix_of.items()},
                    "per_label_acc": {l: metrics(ix, gold, pred)["acc"] for l, ix in ix_label.items()},
                    "pred_dist_by_label": {l: dict(Counter(pred[i] for i in ix)) for l, ix in ix_label.items()},
                    "gaps_acc": {f"{a}-{b}": gap_ci(ix_of[a], ix_of[b], ok, args.n_bootstrap) for a, b in GAPS}}

    group = lambda k: tuple(k.split("/")[::2]) if k.count("/") == 2 else ("-", k)
    mean = lambda xs: float(np.mean([x for x in xs if x is not None] or [np.nan]))
    gaps = [f"{a}-{b}" for a, b in GAPS]
    print(f"\nAccuracy (%), mean over seeds\n{'cond':7}{'model':28}" + "".join(f"{c[:11]:>12}" for c in SETS))
    for g in dict.fromkeys(map(group, out)):
        rs = [v for k, v in out.items() if group(k) == g]
        print(f"{g[0]:7}{g[1]:28}" + "".join(f"{100 * mean([r[c]['acc'] for r in rs]):12.1f}" for c in SETS))
    print("\nGaps (acc, pp), mean over seeds: " + " | ".join(gaps))
    for g in dict.fromkeys(map(group, out)):
        rs = [v for k, v in out.items() if group(k) == g]
        print(f"{g[0]:7}{g[1]:28}" + "".join(f"{100 * mean([r['gaps_acc'][c]['diff'] for r in rs]):+9.1f}" for c in gaps))

    out["_meta"] = {"eval_data": args.eval_data, "sources": srcs, "set_sizes": {k: len(v) for k, v in ix_of.items()}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--model", action="append", metavar="TAG=SRC", help="prediction cache or checkpoint; repeatable")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--conds", nargs="+", default=["cond1", "cond1S", "cond2", "cond2S", "cond3", "cond3S"])
    p.add_argument("--seeds", nargs="+", help="restrict to these seed directories, e.g. seed42")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--output", required=True)
    main(p.parse_args())
