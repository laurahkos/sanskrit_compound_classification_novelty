"""character-level transformer encoder type classifier, and the data, label and metric code shared by all scripts."""
import argparse
import json
import random
import sys
import warnings
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message="enable_nested_tensor")
warnings.filterwarnings("ignore", message="pin_memory")

TYPE_LABELS = ["subord_endo", "subord_exo", "attrib_endo", "attrib_exo", "coord_endo"]
MERGED_LABELS = ["subord_endo", "exo", "attrib_endo", "coord_endo"]

CONLL_FINE_TO_VCD = {
    "T6": "subord_endo", "T3": "attrib_endo", "T4": "subord_endo",
    "T7": "subord_endo", "K7": "attrib_endo", "K6": "attrib_endo",
    "U": "subord_endo", "Tn": "attrib_endo", "Tp": "attrib_endo",
    "K1": "attrib_endo", "Km": "attrib_endo", "Bs6": "subord_exo",
    "BvS": "subord_exo", "Di": "coord_endo", "A1": None,
}
CONLL_TO_VCD = {**CONLL_FINE_TO_VCD, "Tatpurusha": "subord_endo", "Dvandva": "coord_endo",
                "Bahuvrihi": "subord_exo", "Avyayibhava": None}

SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<CLS>", "<SEP>"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


def sacti_label_to_vcd(label):
    return CONLL_TO_VCD.get(label)


def load_jsonl(path):
    """SaCTI records get full_type re-derived from _conll_label, so CONLL_TO_VCD governs every consumer."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_conll_label"):
                r["full_type"] = sacti_label_to_vcd(r["_conll_label"])
            if r.get("full_type") and r.get("compound"):
                records.append(r)
    return records


def detect_and_load(path):
    return load_jsonl(path)


def load_vcd_dir(data_dir):
    return tuple(load_jsonl(Path(data_dir, "type", f"{s}.jsonl")) for s in ("train", "val", "test"))


def load_training_data(args):
    vcd_train, vcd_val, _ = load_vcd_dir(args.vcd_data) if args.vcd_data else ([], [], [])
    conll_train = [r for f in args.conll for r in detect_and_load(f)]
    conll_val = detect_and_load(args.conll_dev) if args.conll_dev else []
    if not vcd_train and not conll_train:
        sys.exit("No training data: provide --vcd-data and/or --conll.")
    return vcd_train, vcd_val, conll_train, conll_val


def read_cache(path, eval_r):
    per_record = json.loads(Path(path).read_text())["per_record"]
    if len(per_record) != len(eval_r) or any(p["gold_type"] != r["full_type"] for p, r in zip(per_record, eval_r)):
        sys.exit(f"{path} does not match the eval file (length or gold labels)")
    return per_record

def scheme(merge_exo):
    """Label set and label map: five classes, or the two exocentric classes scored as one."""
    if merge_exo:
        return MERGED_LABELS, lambda l: "exo" if l.endswith("_exo") else l
    return TYPE_LABELS, lambda l: l


def macro_scores(gold, pred, n):
    """gold, pred: int arrays of class indices < n. Macro averages over gold-active classes."""
    cm = np.bincount(gold * n + pred, minlength=n * n).reshape(n, n)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    p = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
    r = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros(n), where=p + r > 0)
    act = support > 0
    return {"cm": cm, "support": support, "p": p, "r": r, "f1": f1, "acc": tp.sum() / len(gold),
            "macro_p": p[act].mean(), "macro_r": r[act].mean(), "macro_f1": f1[act].mean()}


def score_labels(gold, pred, labels=TYPE_LABELS):
    idx = {l: i for i, l in enumerate(labels)}
    return macro_scores(np.array([idx[g] for g in gold]), np.array([idx[p] for p in pred]), len(labels))


def print_scores(s, title, labels=TYPE_LABELS):
    print(f"\n{title}  n={s['support'].sum()}  acc={100 * s['acc']:.2f}  "
          f"macro P/R/F1={100 * s['macro_p']:.2f}/{100 * s['macro_r']:.2f}/{100 * s['macro_f1']:.2f}")
    for i, l in enumerate(labels):
        if s["support"][i]:
            print(f"  {l:<12} n={s['support'][i]:5d}  P={100 * s['p'][i]:6.2f}  "
                  f"R={100 * s['r'][i]:6.2f}  F1={100 * s['f1'][i]:6.2f}")


def sinusoidal_pe(max_len, dim):
    pos = torch.arange(max_len).unsqueeze(1).float()
    d = torch.arange(0, dim, 2).float()
    pe = torch.zeros(1, max_len, dim)
    pe[0, :, 0::2] = torch.sin(pos / 10000 ** (d / dim))
    pe[0, :, 1::2] = torch.cos(pos / 10000 ** (d / dim))
    return pe


class TypeClassifier(nn.Module):
    def __init__(self, vocab_size, n_types, hidden_dim=256, num_heads=8, num_layers=6,
                 ffn_dim=1024, dropout=0.1, max_len=64, pad_id=0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, n_types=n_types, hidden_dim=hidden_dim, num_heads=num_heads,
                           num_layers=num_layers, ffn_dim=ffn_dim, dropout=dropout, max_len=max_len, pad_id=pad_id)
        H = hidden_dim
        self.embedding = nn.Embedding(vocab_size, H, padding_idx=pad_id)
        self.embed_norm = nn.LayerNorm(H)
        self.embed_drop = nn.Dropout(dropout)
        self.register_buffer("enc_pe", sinusoidal_pe(max_len, H))
        layer = nn.TransformerEncoderLayer(d_model=H, nhead=num_heads, dim_feedforward=ffn_dim,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(H // 2, n_types))

    def forward(self, input_ids, attention_mask):
        x = self.embed_norm(self.embedding(input_ids))
        x = self.embed_drop(x + self.enc_pe[:, :x.size(1)])
        h = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        m = attention_mask.unsqueeze(-1)
        return self.classifier((h * m).sum(1) / m.sum(1).float().clamp(min=1))


def load_model(checkpoint, device):
    d = Path(checkpoint).parent
    cfg = json.loads((d / "config.json").read_text())
    use_sentence = cfg.pop("use_sentence", False)
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    model = TypeClassifier(**cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict({k: v for k, v in state.items() if not k.endswith("enc_pe")}, strict=False)
    return model.eval(), vocab, use_sentence

def build_vocab(records, use_sentence):
    chars = set()
    for r in records:
        chars.update(r["compound"])
        if use_sentence:
            chars.update(r.get("sentence", "").strip())
    return {s: i for i, s in enumerate(SPECIALS + sorted(chars))}

def encode_ids(r, vocab, max_len, use_sentence):
    unk = vocab["<UNK>"]
    ids = [vocab["<CLS>"]] + [vocab.get(c, unk) for c in r["compound"]] + [vocab["<SEP>"]]
    sentence = r.get("sentence", "").strip() if use_sentence else ""
    budget = max_len - len(ids) - 1
    if budget > 0 and sentence:
        ids += [vocab.get(c, unk) for c in sentence[:budget]] + [vocab["<SEP>"]]
    return ids[:max_len]

def encode(records, vocab, max_len, use_sentence):
    return [(torch.tensor(encode_ids(r, vocab, max_len, use_sentence)), TYPE_LABELS.index(r["full_type"]))
            for r in records]


def collate(batch):
    ids = nn.utils.rnn.pad_sequence([x for x, _ in batch], batch_first=True)
    return ids, (ids != 0).long(), torch.tensor([y for _, y in batch])


class MixedBatchIterator:
    def __init__(self, loader_a, loader_b, mix_ratio):
        self.loader_a, self.loader_b, self.mix_ratio = loader_a, loader_b, mix_ratio

    def __len__(self):
        return max(len(self.loader_a), len(self.loader_b))

    def __iter__(self):
        a, b = cycle(self.loader_a), cycle(self.loader_b)
        for _ in range(len(self)):
            yield next(b) if random.random() < self.mix_ratio else next(a)

def build_lookup(records, threshold=0.60):
    pair_types = defaultdict(Counter)
    for r in records:
        l1, l2 = r.get("lemma1", "").strip(), r.get("lemma2", "").strip()
        if l1 and l2:
            pair_types[(l1, l2)][r["full_type"]] += 1
    lookup = {}
    for pair, counts in pair_types.items():
        best, n = counts.most_common(1)[0]
        if n / sum(counts.values()) >= threshold:
            lookup[pair] = best
    return lookup


def class_weights(records, scheme_name):
    counts = Counter(r["full_type"] for r in records)
    n, k = sum(counts.values()), len(TYPE_LABELS)

    def weight(c):
        if c == 0:
            return 1.0
        if scheme_name == "inverse":
            return n / (k * c)
        if scheme_name == "sqrt-inverse":
            return (n / (k * c)) ** 0.5
        beta = (n - 1) / n
        return (1 - beta) / (1 - beta ** c)

    w = torch.tensor([weight(counts[t]) for t in TYPE_LABELS])
    return w / w.mean()


@torch.no_grad()
def predict_ids(model, loader):
    model.eval()
    device = next(model.parameters()).device
    return [i for ids, mask, _ in loader for i in model(ids.to(device), mask.to(device)).argmax(-1).tolist()]


def train_classifier(vcd_train, vcd_val, conll_train, conll_val, out_dir, args, device):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab(vcd_train + vcd_val + conll_train + conll_val, args.use_sentence)
    train = vcd_train + conll_train
    weights = None if args.class_weights == "none" else class_weights(train, args.class_weights).to(device)
    lookup = build_lookup(vcd_train)
    val = vcd_val + conll_val
    if not val:
        train, val = train_test_split(train, test_size=0.05, random_state=args.seed)
        vcd_train = [r for r in train if r.get("_source") != "conll"]
        conll_train = [r for r in train if r.get("_source") == "conll"]

    def loader(records, shuffle=True, batch_size=args.batch_size):
        return DataLoader(encode(records, vocab, args.max_len, args.use_sentence),
                          batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    val_loader = loader(val, False, 2 * args.batch_size)
    train_iter = (MixedBatchIterator(loader(vcd_train), loader(conll_train), args.mix_ratio)
                  if vcd_train and conll_train else loader(vcd_train or conll_train))
    model = TypeClassifier(len(vocab), len(TYPE_LABELS), args.hidden_dim, args.num_heads, args.num_layers,
                           args.ffn_dim, args.dropout, args.max_len, vocab["<PAD>"]).to(device)
    (out_dir / "config.json").write_text(json.dumps({**model.config, "use_sentence": args.use_sentence}, indent=2))
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    if lookup:
        (out_dir / "lookup.json").write_text(
            json.dumps({f"{a}|{b}": t for (a, b), t in lookup.items()}, ensure_ascii=False, indent=2))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    warmup = max(1, int(len(train_iter) * args.warmup_epochs))
    val_gold = [r["full_type"] for r in val]
    step, best, best_epoch, bad, history = 0, -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, mask, labels in train_iter:
            if step < warmup:
                for g in optimizer.param_groups:
                    g["lr"] = args.lr * (step + 1) / warmup
            loss = F.cross_entropy(model(ids.to(device), mask.to(device)), labels.to(device),
                                   weight=weights, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            losses.append(loss.item())
        f1 = float(score_labels(val_gold, [TYPE_LABELS[i] for i in predict_ids(model, val_loader)])["macro_f1"])
        print(f"epoch {epoch:3d}  loss={np.mean(losses):.3f}  val F1={f1:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(f1)
        history.append((epoch, f1))
        if f1 > best:
            best, best_epoch, bad = f1, epoch, 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break

    (out_dir / "trainer_state.json").write_text(json.dumps({
        "selection_metric": "val_macro_f1", "best_metric": best, "best_epoch": best_epoch,
        "final_epoch": epoch, "max_epochs": args.epochs, "batch_size": args.batch_size,
        "patience": args.patience, "early_stopped": bad >= args.patience, "val_macro_f1_by_epoch": history,
    }, indent=2))
    return best


def evaluate_records(model, records, vocab, lookup=None, use_sentence=False, batch_size=64):
    loader = DataLoader(encode(records, vocab, model.config["max_len"], use_sentence),
                        batch_size=batch_size, collate_fn=collate)
    per_record = []
    for r, i in zip(records, predict_ids(model, loader)):
        key = (r.get("lemma1", "").strip(), r.get("lemma2", "").strip())
        pred, src = (lookup[key], "lookup") if lookup and key in lookup else (TYPE_LABELS[i], "neural")
        per_record.append({"compound": r.get("compound", ""), "gold_type": r["full_type"],
                           "pred_type": pred, "src": src})
    return per_record

def add_common_args(p):
    p.add_argument("--vcd-data", help="directory with type/{train,val,test}.jsonl")
    p.add_argument("--conll", nargs="+", default=[])
    p.add_argument("--conll-dev")
    p.add_argument("--mix-ratio", type=float, default=0.5, help="probability of drawing SaCTI in joint training")
    p.add_argument("--use-sentence", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)


def add_classifier_args(p, prefix=""):
    p.add_argument(f"--{prefix}class-weights", default="sqrt-inverse",
                   choices=["none", "inverse", "sqrt-inverse", "effective"])
    for name, default in [("max-len", 64), ("hidden-dim", 256), ("num-heads", 8), ("num-layers", 6),
                          ("ffn-dim", 1024), ("dropout", 0.1), ("epochs", 40), ("batch-size", 64),
                          ("lr", 1e-4), ("warmup-epochs", 2.0), ("patience", 8)]:
        p.add_argument(f"--{prefix}{name}", type=type(default), default=default)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_common_args(p)
    add_classifier_args(p)
    args = p.parse_args()
    set_seed(args.seed)
    train_classifier(*load_training_data(args), args.output, args, get_device())
