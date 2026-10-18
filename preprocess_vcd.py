from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from type_classifier import TYPE_LABELS


def make_vcd_type(cpd_type: str, endo_exo: str) -> str | None:
    """cpdType + endoExo -> one of TYPE_LABELS, or None if unmapped."""
    ct = str(cpd_type).strip().lower()
    ee = str(endo_exo).strip().lower()
    if ct == "coord":
        return "coord_endo"
    full = f"{ct}_{ee}"
    return full if full in TYPE_LABELS else None


def fix(s) -> str:
    return str(s).replace("\uffb1", "ḷ").strip()


def build_vcd_type_records(csv_path: str) -> list[dict]:
    """Reads VCD-final.csv into load_jsonl()/load_vcd_dir()'s schema.
    intLabel/pos1/pos2 kept only as _*_raw for provenance — never encoded
    into any model input (encode_ids()/format_input() use compound [+
    sentence] only)."""
    import pandas as pd
    df = pd.read_csv(csv_path, sep="\t", on_bad_lines="skip", quoting=1)
    records = []
    for _, row in df.iterrows():
        compound  = fix(row["compound"])
        full_type = make_vcd_type(row.get("cpdType", ""), row.get("endoExo", ""))
        if not compound or full_type is None:
            continue
        records.append({
            "compound":  compound,
            "lemma1": fix(row["lemma1"]),
            "lemma2": fix(row["lemma2"]),
            "full_type": full_type,
            "veda":  str(row.get("veda", "")),
            "register": str(row.get("register", "")),
            "sentence":  fix(row.get("sentence", "")),
            "cpdID":  str(row.get("cpdID", "")).strip(),
            "_intlabel_raw": str(row.get("intLabel", "")).strip(),
            "_pos1_raw": str(row.get("pos1", "")).strip(),
            "_pos2_raw":  str(row.get("pos2", "")).strip(),
        })
    return records


def write_vcd_type_splits(csv_path: str, output_dir: str,
                          val_size: float = 0.1, test_size: float = 0.1,
                          seed: int = 42) -> None:
    """Writes <output_dir>/type/{train,val,test}.jsonl (NESTED layout)."""
    from sklearn.model_selection import train_test_split

    records = build_vcd_type_records(csv_path)
    print(f"  Loaded {len(records)} usable VCD type records from {csv_path}")
    dist = Counter(r["full_type"] for r in records)
    print(f"  Class distribution: {dict(dist)}")

    strata = [r["full_type"] for r in records]
    try:
        tr_val, te = train_test_split(records, test_size=test_size,
                                      random_state=seed, stratify=strata)
        strata_tv = [r["full_type"] for r in tr_val]
        tr, va = train_test_split(tr_val, test_size=val_size / (1 - test_size),
                                  random_state=seed, stratify=strata_tv)
    except ValueError as e:
        print(f"  WARNING: stratified split failed ({e}); falling back to "
             f"unstratified split. Class proportions across splits are NOT "
             f"guaranteed in this fallback case.")
        tr_val, te = train_test_split(records, test_size=test_size, random_state=seed)
        tr, va = train_test_split(tr_val, test_size=val_size / (1 - test_size),
                                  random_state=seed)

    out = Path(output_dir) / "type"
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", tr), ("val", va), ("test", te)]:
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Wrote {path}  (n={len(split)})")

    print(f"\n  seed={seed}  val_size={val_size}  test_size={test_size}  "
         f"(re-run with the same seed to reproduce this split exactly)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to VCD-final.csv")
    p.add_argument("--output", required=True,
                   help="Output directory (writes <output>/type/*.jsonl)")
    p.add_argument("--val-size",  type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int,  default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_vcd_type_splits(args.input, args.output,
                          val_size=args.val_size, test_size=args.test_size,
                          seed=args.seed)
