import argparse
import json
from pathlib import Path

import numpy as np

from stratify_eval import categorize, strata, train_rubric
from type_classifier import TYPE_LABELS, detect_and_load, macro_scores, read_cache, scheme

TAGS = {"classifier": "clf", "google_byt5-base": "base", "buddhist-nlp_byt5-sanskrit": "skt"}
N = len(TYPE_LABELS)


def arrays(per_record, merge_exo):
    labels, norm = scheme(merge_exo)
    idx = {l: i for i, l in enumerate(labels)}
    return (np.array([idx[norm(p["gold_type"])] for p in per_record]),
            np.array([idx[norm(p["pred_type"])] for p in per_record]))


def load_system(cache_root, system, dataset, eval_r, merge_exo=False):
    cond, tag = system.split(":", 1)
    files = sorted(Path(cache_root, cond).glob(f"seed*/{tag}_{dataset}.json"))
    if not files:
        raise FileNotFoundError(f"no caches for {system} under {cache_root}/{cond}/seed*/{tag}_{dataset}.json")
    return {f.parent.name: arrays(read_cache(f, eval_r), merge_exo) for f in files}


def f1(g, p):
    return macro_scores(g, p, N)["macro_f1"]


def acc(g, p):
    return (g == p).mean()


def seed_summary(runs, ix):
    out = {"seeds": list(runs)}
    for name, fn in (("macro_f1", f1), ("acc", acc)):
        xs = [float(fn(g[ix], p[ix])) for g, p in runs.values()]
        out.update({name: xs, f"{name}_mean": float(np.mean(xs)),
                    f"{name}_sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None})
    return out


def two_level_ci(runs_a, runs_b, ix, n_boot, rng, alpha=0.05):
    A, B, ix = list(runs_a.values()), list(runs_b.values()), np.asarray(ix)
    boot = {"macro_f1": np.empty(n_boot), "acc": np.empty(n_boot)}
    for k in range(n_boot):
        sa, sb = rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B))
        it = ix[rng.integers(0, len(ix), len(ix))]
        for name, fn in (("macro_f1", f1), ("acc", acc)):
            boot[name][k] = (np.mean([fn(A[s][0][it], A[s][1][it]) for s in sa])
                             - np.mean([fn(B[s][0][it], B[s][1][it]) for s in sb]))
    a, b = seed_summary(runs_a, ix), seed_summary(runs_b, ix)
    out = {"a": a, "b": b}
    for name, d in boot.items():
        obs = a[f"{name}_mean"] - b[f"{name}_mean"]
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p = min(1.0, 2 * ((d <= 0).mean() if obs >= 0 else (d >= 0).mean()))
        out[name] = {"diff": obs, "ci": [float(lo), float(hi)], "p": float(p)}
    return out


def report(name, n, r):
    sd = lambda s: 100 * (s["macro_f1_sd"] or 0)
    f, a = r["macro_f1"], r["acc"]
    print(f"  {name:12} n={n:5}  F1 {100 * r['a']['macro_f1_mean']:5.1f}±{sd(r['a']):.1f} "
          f"vs {100 * r['b']['macro_f1_mean']:5.1f}±{sd(r['b']):.1f}  "
          f"ΔF1 {100 * f['diff']:+5.1f} [{100 * f['ci'][0]:+.1f}, {100 * f['ci'][1]:+.1f}] p={f['p']:.3f}  "
          f"Δacc {100 * a['diff']:+5.1f} [{100 * a['ci'][0]:+.1f}, {100 * a['ci'][1]:+.1f}]")


def main(args):
    eval_r = detect_and_load(args.eval_data)
    groups = {"overall": list(range(len(eval_r)))}
    if args.train_data:
        s = strata(categorize(eval_r, args.dataset, train_rubric(args.train_data)))
        groups |= {c: s[c] for c in ["1.1", "1.2", "2", "0", "1.2a", "1.2b"] if s[c]}  # order fixes the RNG stream
    rng = np.random.default_rng(args.seed)
    out = {"dataset": args.dataset, "eval_data": args.eval_data, "train_data": args.train_data,
           "merge_exo": args.merge_exo, "n_bootstrap": args.n_bootstrap,
           "strata_n": {k: len(v) for k, v in groups.items()}, "pairs": {}}
    for a, b in args.pair:
        ra = load_system(args.cache_root, a, args.dataset, eval_r, args.merge_exo)
        rb = load_system(args.cache_root, b, args.dataset, eval_r, args.merge_exo)
        print(f"\n{a} ({len(ra)} seeds) vs {b} ({len(rb)} seeds)")
        res = out["pairs"][f"{a} vs {b}"] = {}
        for name, ix in groups.items():
            res[name] = two_level_ci(ra, rb, ix, args.n_bootstrap, rng)
            report(name, len(ix), res[name])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--dataset", required=True, choices=["vcd", "sacti"])
    p.add_argument("--eval-data", required=True)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("COND:TAG", "COND:TAG"))
    p.add_argument("--train-data", nargs="+", help="training sources defining the novelty strata")
    p.add_argument("--merge-exo", action="store_true",
                   help="score the two exocentric classes as one; use for cross-corpus comparisons "
                        "(SaCTI has no attrib_exo, so SaCTI-trained models cap at 80 five-class macro-F1 on VCD)")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    main(p.parse_args())
