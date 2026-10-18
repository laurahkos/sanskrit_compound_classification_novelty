"""main.md overall scores; novelty.md macro-F1 per novelty category; tags.md SaCTI per-tag accuracy;
confusions.md prediction distribution per tag; t3_split.md T3 by licensing sūtra; errors.md misclassified items."""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from seed_compare import TAGS
from stratify_eval import CATS
from type_classifier import detect_and_load, read_cache

CONDS = ["cond1", "cond1S", "cond2", "cond2S", "cond3", "cond3S", "cond3B", "cond3BS"]
SCHEMES = [("vcd", "5class"), ("vcd", "merged"), ("sacti", "merged")]
LABEL_ORDER = ["T6", "T4", "T7", "T3", "U", "Tn", "Tp", "K1", "K6", "K7", "Km", "Bs6", "BvS", "Di"]
CLASSES = ["subord_endo", "attrib_endo", "subord_exo", "attrib_exo", "coord_endo"]


def ms(xs):
    xs = [100 * x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{np.mean(xs):.1f}" if len(xs) < 2 else f"{np.mean(xs):.1f}±{np.std(xs, ddof=1):.1f}"


def row(*cells):
    return "| " + " | ".join(map(str, cells)) + " |"


def header(*cells):
    return [row(*cells), "|" + "---|" * len(cells)]


def write(path, lines):
    path.write_text("\n".join(lines) + "\n")


def seeds(root, cond):
    return sorted(p.name for p in Path(root, cond).glob("seed*") if p.is_dir())


def strat_runs(root, cond, tag, dom, scheme):
    files = (Path(root, cond, s, f"{tag}_{dom}_{scheme}.json") for s in seeds(root, cond))
    return [json.loads(f.read_text()) for f in files if f.exists()]


def sacti_caches(root, cond, tag, sacti):
    files = ((s, Path(root, cond, s, f"{tag}_sacti.json")) for s in seeds(root, cond))
    return [(s, read_cache(f, sacti)) for s, f in files if f.exists()]


def score_tables(args, out):
    main = ["# Overall macro-F1 (acc), mean ± sd over seeds"]
    nov = ["# Macro-F1 by novelty category, mean ± sd over seeds (n = items in category)"]
    for dom, scheme in SCHEMES:
        main += [f"\n## {dom.upper()} test, {scheme}\n", *header("cond", "model", "macro-F1", "acc", "seeds")]
        nov += [f"\n## {dom.upper()} test, {scheme}\n", *header("cond", "model", *CATS)]
        for cond in CONDS:
            counts = None
            for tag, short in TAGS.items():
                rs = strat_runs(args.stratify_root, cond, tag, dom, scheme)
                if not rs:
                    continue
                counts = rs[0]["category_counts"]
                main.append(row(cond, short, ms([r["overall_macro_f1"] for r in rs]),
                                ms([r["overall_accuracy"] for r in rs]), len(rs)))
                nov.append(row(cond, short, *(ms([r["stratified_results"][c]["macro_f1"] for r in rs
                                                  if r["stratified_results"].get(c, {}).get("n")]) for c in CATS)))
            if counts:
                nov.append(row(cond, "*n*", *(counts.get(c, 0) for c in CATS)))
    write(out / "main.md", main)
    write(out / "novelty.md", nov)


def karaka_tables(args, out):
    acc, dist = defaultdict(list), defaultdict(Counter)
    for cond in CONDS:
        for seed in seeds(args.karaka_root, cond):
            f = Path(args.karaka_root, cond, seed, "karaka_axis.json")
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            for tag, short in TAGS.items():
                for lab, a in d.get(tag, {}).get("per_label_acc", {}).items():
                    acc[(cond, short, lab)].append(a)
                for lab, c in d.get(tag, {}).get("pred_dist_by_label", {}).items():
                    dist[(cond, short, lab)].update(c)
    labels = [l for l in LABEL_ORDER if any(k[2] == l for k in acc)]
    lines = ["# SaCTI per-tag accuracy (%), mean ± sd over seeds\n", *header("cond", "model", *labels)]
    for cond in CONDS:
        for short in TAGS.values():
            if any((cond, short, l) in acc for l in labels):
                lines.append(row(cond, short, *(ms(acc.get((cond, short, l), [])) for l in labels)))
    write(out / "tags.md", lines)

    lines = ["# Where each tag's predictions go (% of items, pooled over seeds)"]
    for lab in ["T3", "T7", "U", "K1", "K6"]:
        lines += [f"\n## {lab}\n", *header("cond", "model", *CLASSES)]
        for cond in CONDS:
            for short in TAGS.values():
                c = dist.get((cond, short, lab))
                if c:
                    lines.append(row(cond, short, *(f"{100 * c[k] / sum(c.values()):.0f}" for k in CLASSES)))
    write(out / "confusions.md", lines)


# Approximate T3 split by licensing sūtra, matched on the second member (check the listed items by hand).
# A 2.1.31 pūrvasadṛśasamonārthakalahanipuṇamiśraślakṣṇaiḥ: closed list (ūnārtha incl. hīna, rahita, śūnya ...).
# A 2.1.32 kartṛkaraṇe kṛtā bahulam: kta participle head (stem in -ta/-na/-ṇa/-ṭa/-ḍha/-dha, not -ana nouns).
LIST_2131 = ("pūrv", "sadṛś", "sam", "ūn", "nyūn", "hīn", "rahit", "vikal", "śūny", "varjit",
             "kalah", "nipuṇ", "miśr", "ślakṣṇ")
ENDINGS = sorted(["aḥ", "am", "aṃ", "ā", "āḥ", "ena", "asya", "āya", "āt", "e", "au", "ān", "āni", "aiḥ",
                  "ebhyaḥ", "eṣu", "ānām", "ām", "ayā", "āyāḥ", "āyām", "āyai", "o", "a"], key=len, reverse=True)
PART_STEM = re.compile(r"(t|n|ṇ|ṭ|ḍh|dh)$")
GROUPS = ["2.1.32", "2.1.31", "other"]


def is_participle(word):
    e = next((e for e in ENDINGS if word.endswith(e) and len(word) > len(e) + 1), None)
    if e is None:
        return False
    stem = word[:-len(e)]
    return not stem.endswith(("an", "ān")) and bool(PART_STEM.search(stem))


def t3_group(r):
    cpd = r.get("compound", "")
    head = cpd.split("-", 1)[1] if "-" in cpd else ""
    if head.startswith(LIST_2131) and not head.startswith(("samā", "saṃ", "samp", "samy")):
        return "2.1.31"
    return "2.1.32" if is_participle(head) else "other"


def t3_split(args, out, sacti):
    idx = [i for i, r in enumerate(sacti) if r.get("_conll_label") == "T3"]
    grp = {i: t3_group(sacti[i]) for i in idx}
    members = {g: [i for i in idx if grp[i] == g] for g in GROUPS}
    lines = [f"# T3 by licensing sūtra (approximate; n = {len(idx)}: "
             + ", ".join(f"{g} {len(members[g])}" for g in GROUPS) + ")\n",
             "2.1.31 = second member from the sūtra's list; 2.1.32 = other participle heads; other = neither.\n",
             *header("cond", "model", *(f"acc {g}" for g in GROUPS), *(f"→subord_endo {g}" for g in GROUPS), "seeds")]
    to_subord = Counter()
    for cond in CONDS:
        for tag, short in TAGS.items():
            runs = sacti_caches(args.cache_root, cond, tag, sacti)
            if not runs:
                continue
            share = lambda g, label: [np.mean([pr[i]["pred_type"] == label for i in members[g]])
                                      if members[g] else None for _, pr in runs]
            lines.append(row(cond, short, *(ms(share(g, "attrib_endo")) for g in GROUPS),
                             *(ms(share(g, "subord_endo")) for g in GROUPS), len(runs)))
            if cond in ("cond2", "cond3"):
                to_subord.update(i for _, pr in runs for i in idx if pr[i]["pred_type"] == "subord_endo")
    lines += ["\n## T3 items most often predicted subord_endo in-domain (cond2/cond3, all models and seeds)\n",
              *header("compound", "group", "times → subord_endo")]
    lines += [row(sacti[i].get("compound", ""), grp[i], c) for i, c in to_subord.most_common(30)]
    lines.append("\n## Items per group (2.1.31 in full; random samples of the others)\n")
    rng = np.random.default_rng(0)
    for g in GROUPS:
        pool = members[g]
        pick = pool if g == "2.1.31" or not pool else rng.choice(pool, size=min(20, len(pool)), replace=False)
        lines.append(f"**{g}** ({len(pool)}): " + ", ".join(sacti[i].get("compound", "") for i in pick) + "\n")
    write(out / "t3_split.md", lines)


def error_samples(args, out, sacti, cond="cond3", tag="buddhist-nlp_byt5-sanskrit", per_tag=6):
    runs = sacti_caches(args.cache_root, cond, tag, sacti)
    if not runs:
        return
    seed, pr = runs[0]
    by_tag = defaultdict(list)
    for r, p in zip(sacti, pr):
        if p["pred_type"] != p["gold_type"]:
            by_tag[r.get("_conll_label")].append(row(r.get("_conll_label"), r.get("compound", ""),
                                                     p["gold_type"], p["pred_type"]))
    write(out / "errors.md", [f"# Misclassified SaCTI test items ({cond}, {tag}, {seed}), up to {per_tag} per tag\n",
                              *header("tag", "compound", "gold", "predicted"),
                              *(line for lab in LABEL_ORDER for line in by_tag[lab][:per_tag])])


def main(args):
    found = {"stratify": list(Path(args.stratify_root).glob("cond*/seed*/*.json")),
             "karaka": list(Path(args.karaka_root).glob("cond*/seed*/karaka_axis.json")),
             "cache": list(Path(args.cache_root).glob("cond*/seed*/*_sacti.json"))}
    for k, v in found.items():
        print(f"{k:9} {len(v):4d} files under {getattr(args, k + '_root')}")
    if not all(found.values()):
        sys.exit(f"no input files for {[k for k, v in found.items() if not v]}; check the --*-root arguments")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    score_tables(args, out)
    karaka_tables(args, out)
    sacti = detect_and_load(args.sacti_test)
    t3_split(args, out, sacti)
    error_samples(args, out, sacti)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stratify-root", default="posthoc/stratify")
    p.add_argument("--karaka-root", default="karaka")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--sacti-test", default="data/sacti_dedup_context/test.jsonl")
    p.add_argument("--out", default="posthoc/summary")
    main(p.parse_args())
