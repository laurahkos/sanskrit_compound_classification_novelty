import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from type_classifier import sacti_label_to_vcd

VIRAMA = "्"
CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ", "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h", "ळ": "ḷ",
}
MATRAS = {"ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū", "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ",
          "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o"}
SIGNS = {"अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū", "ऋ": "ṛ", "ॠ": "ṝ", "ऌ": "ḷ",
         "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ं": "ṃ", "ँ": "ṁ", "ः": "ḥ", ":": "ḥ", "ऽ": "'", "ऑ": "o", "ॐ": "oṃ", VIRAMA: "", **MATRAS,
         **dict(zip("०१२३४५६७८९", "0123456789"))}


def unmojibake(m):
    return b"".join(c.encode("latin-1" if ord(c) < 256 else "cp1252") for c in m.group()).decode("utf-8")


def deva_to_iast(text):
    text = re.sub("à[¤¥].", unmojibake, text)
    text = unicodedata.normalize("NFD", text).replace("\u093c", "")
    out, i = [], 0
    while i < len(text):
        c, nxt = text[i], text[i + 1:i + 2]
        i += 1
        if c in CONSONANTS:
            out.append(CONSONANTS[c])
            if nxt == VIRAMA:
                i += 1
            elif nxt in MATRAS:
                out.append(MATRAS[nxt])
                i += 1
            else:
                out.append("a")
        else:
            out.append(SIGNS.get(c, c))
    return "".join(out)


def load_conll(path):
    records, block, n_blocks = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                if "\t" in line:
                    block.append(line.split("\t"))
                continue
            if not block:
                continue
            root = next((row for row in block if row[6] == "0"), None)
            full_type = root and sacti_label_to_vcd(root[7])
            if full_type:
                compound = deva_to_iast(root[1])
                if compound:
                    records.append({"compound": compound, "full_type": full_type,
                                    "_source": "conll", "_conll_label": root[7],
                                    "sentence": deva_to_iast(" ".join(row[1] for row in block)),
                                    "sent_id": f"block{n_blocks}"})
            block = []
            n_blocks += 1
    return records


def pool_and_dedup(conll_paths: list):
    """Returns (kept, dropped): kept = one record per distinct (compound,
    sentence) pair with _n_occurrences added; dropped = pairs whose label
    disagrees across occurrences of the identical (compound, sentence)."""
    by_key = defaultdict(list)
    for path in conll_paths:
        for r in load_conll(path):
            by_key[(r["compound"], r.get("sentence", ""))].append(r)

    kept, dropped = [], []
    for key, occurrences in by_key.items():
        labels = {r["_conll_label"] for r in occurrences}
        if len(labels) > 1:
            dropped.append((key, labels))
            continue
        rep = dict(occurrences[0])
        rep["_n_occurrences"] = len(occurrences)
        kept.append(rep)

    return kept, dropped


def stratified_split(records: list, train_frac: float, dev_frac: float,
                     test_frac: float, seed: int):
    assert abs(train_frac + dev_frac + test_frac - 1.0) < 1e-6
    by_class = defaultdict(list)
    for r in records:
        by_class[r["full_type"]].append(r)

    rng = random.Random(seed)
    train, dev, test = [], [], []
    for recs in by_class.values():
        recs = recs[:]
        rng.shuffle(recs)
        n = len(recs)
        n_train = round(n * train_frac)
        n_dev = round(n * dev_frac)
        train += recs[:n_train]
        dev += recs[n_train:n_train + n_dev]
        test += recs[n_train + n_dev:]

    rng.shuffle(train); rng.shuffle(dev); rng.shuffle(test)
    return train, dev, test


def write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(args):
    kept, dropped = pool_and_dedup(args.conll)
    out_dir = Path(args.output_dir)
    write_jsonl([{"compound": c, "sentence": s, "labels": sorted(l)} for (c, s), l in dropped],
                out_dir / "dropped_ambiguous.jsonl")
    train, dev, test = stratified_split(kept, args.train_frac, args.dev_frac, args.test_frac, args.seed)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(dev, out_dir / "dev.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    print(f"kept={len(kept)} dropped={len(dropped)} train={len(train)} dev={len(dev)} test={len(test)}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conll", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--dev-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
