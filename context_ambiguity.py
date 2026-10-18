import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from seed_compare import TAGS, load_system, report, two_level_ci
from stratify_eval import clean
from type_classifier import detect_and_load, scheme


def surface(r):
    """The compound without the member hyphen, as the old surface field."""
    return clean(r["compound"].replace("-", ""))


def main(args):
    _, norm = scheme(args.merge_exo)
    labels_of, fine_of = defaultdict(set), defaultdict(Counter)
    for path in args.pool:
        for r in detect_and_load(path):
            labels_of[surface(r)].add(norm(r["full_type"]))
            if r.get("_conll_label"):
                fine_of[surface(r)][r["_conll_label"]] += 1

    eval_r = detect_and_load(args.eval_data)
    amb = [i for i, r in enumerate(eval_r) if len(labels_of[surface(r)]) > 1]
    unamb = sorted(set(range(len(eval_r))) - set(amb))
    print(f"type-ambiguous test items: {len(amb)}/{len(eval_r)}, "
          f"{len({surface(eval_r[i]) for i in amb})} distinct strings")
    if len(amb) < 30:
        print("WARNING: fewer than 30 ambiguous items; report counts and examples, not significance.")

    top = Counter(surface(eval_r[i]) for i in amb).most_common(args.n_examples)
    rng = np.random.default_rng(args.seed)
    out = {"dataset": args.dataset, "merge_exo": args.merge_exo, "pool": args.pool,
           "n_eval": len(eval_r), "n_ambiguous": len(amb), "comparisons": {},
           "examples": [{"surface": s, "n_test": c, "labels": sorted(labels_of[s]), "fine_tags": dict(fine_of[s])}
                        for s, c in top]}
    for cond in args.conds:
        for tag in TAGS:
            a, b = f"{cond}S:{tag}", f"{cond}:{tag}"
            try:
                ra = load_system(args.cache_root, a, args.dataset, eval_r, args.merge_exo)
                rb = load_system(args.cache_root, b, args.dataset, eval_r, args.merge_exo)
            except FileNotFoundError as e:
                print(f"skip: {e}")
                continue
            print(f"\n{a} vs {b} (context minus no context)")
            res = out["comparisons"][f"{a} vs {b}"] = {}
            for name, ix in (("ambiguous", amb), ("unambiguous", unamb)):
                if len(ix) >= 2:
                    res[name] = two_level_ci(ra, rb, ix, args.n_bootstrap, rng)
                    report(name, len(ix), res[name])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--dataset", required=True, choices=["vcd", "sacti"])
    p.add_argument("--eval-data", required=True)
    p.add_argument("--pool", nargs="+", required=True, help="all splits: ambiguity is judged over the whole corpus")
    p.add_argument("--conds", nargs="+", default=["cond1", "cond2", "cond3"])
    p.add_argument("--merge-exo", action="store_true")
    p.add_argument("--n-examples", type=int, default=15)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    main(p.parse_args())
