"""Paired item bootstrap between two single runs (caches or checkpoints) on the same eval file."""
import argparse
import json
from pathlib import Path
import numpy as np
from cache_predictions import load_predictions
from seed_compare import arrays, report, two_level_ci
from stratify_eval import categorize, strata, train_rubric
from type_classifier import detect_and_load

def main(args):
    eval_r = detect_and_load(args.eval_data)
    ra, rb = ({"run": arrays(load_predictions(src, eval_r, args.batch_size, args.use_lookup), args.merge_exo)}
              for src in (args.a, args.b))
    groups = {"overall": list(range(len(eval_r)))}
    if args.train_data:
        groups |= {c: ix for c, ix in strata(categorize(eval_r, args.dataset, train_rubric(args.train_data))).items()
                   if ix}
    rng = np.random.default_rng(args.seed)
    out = {"a": args.a, "b": args.b, "eval_data": args.eval_data, "n_bootstrap": args.n_bootstrap, "results": {}}
    for name, ix in groups.items():
        out["results"][name] = two_level_ci(ra, rb, ix, args.n_bootstrap, rng)
        report(name, len(ix), out["results"][name])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="prediction cache (.json) or checkpoint")
    p.add_argument("--b", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--dataset", choices=["vcd", "sacti"], default="vcd", help="identity scheme for the strata")
    p.add_argument("--train-data", nargs="+", help="training sources defining the novelty strata")
    p.add_argument("--merge-exo", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--use-lookup", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    main(p.parse_args())
