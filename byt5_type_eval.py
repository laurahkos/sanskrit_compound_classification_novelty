"""Fine-tune ByT5 models for compound type"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback)
from transformers.modeling_outputs import BaseModelOutput
from type_classifier import (TYPE_LABELS, add_classifier_args, add_common_args, get_device, load_training_data,
                             score_labels, set_seed, train_classifier)

MODEL_NAMES = {"byt5-base": "google/byt5-base", "byt5-sanskrit": "buddhist-nlp/byt5-sanskrit"}

def format_input(r, use_sentence):
    sentence = (r.get("sentence") or "").strip() if use_sentence else ""
    return f"{r['compound']} | context: {sentence}" if sentence else r["compound"]

class Seq2SeqTypeDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, use_sentence):
        self.records, self.tokenizer, self.max_len, self.use_sentence = records, tokenizer, max_len, use_sentence

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        src = self.tokenizer(format_input(r, self.use_sentence), max_length=self.max_len, truncation=True)
        tgt = self.tokenizer(r["full_type"], max_length=16, truncation=True)
        return {"input_ids": src["input_ids"], "attention_mask": src["attention_mask"], "labels": tgt["input_ids"]}

@torch.no_grad()
def label_logprobs(model, tokenizer, texts, max_len, cand_ids, cand_mask):
    """(B, L) summed log P(label | input) for each candidate label, tokenized as in training (bytes + </s>)."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
    B, L = len(texts), cand_ids.size(0)
    h = model.get_encoder()(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    tgt, tmask = cand_ids.repeat(B, 1), cand_mask.repeat(B, 1)
    logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=h.repeat_interleave(L, 0)),
                   attention_mask=enc["attention_mask"].repeat_interleave(L, 0),
                   labels=tgt.masked_fill(tmask == 0, -100)).logits
    lp = logits.float().log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * tmask
    return lp.sum(-1).view(B, L)

def evaluate_records_seq2seq(model, tokenizer, records, batch_size=32, use_sentence=False, max_len=64):
    model.eval()
    cand = tokenizer(TYPE_LABELS, padding=True, return_tensors="pt").to(model.device)
    per_record = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        scores = label_logprobs(model, tokenizer, [format_input(r, use_sentence) for r in batch], max_len,
                                cand["input_ids"], cand["attention_mask"]).cpu()
        for r, sc, k in zip(batch, scores.tolist(), scores.argmax(-1).tolist()):
            per_record.append({"compound": r["compound"], "gold_type": r["full_type"],
                               "pred_type": TYPE_LABELS[k],
                               "label_logprobs": {l: round(s, 4) for l, s in zip(TYPE_LABELS, sc)}})
    return per_record

class CountingSampler(Sampler):
    def __init__(self, inner, sources, feed):
        self.inner, self.sources, self.feed = inner, sources, feed

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        for i in self.inner:
            self.feed["draws"][self.sources[i]] += 1
            self.feed["seen"][self.sources[i]].add(i)
            yield i

class TrackingTrainer(Seq2SeqTrainer):
    """Optionally draws examples with replacement from per-example weights; counts what is fed per source."""

    def __init__(self, *a, sample_weights, sources, feed, **kw):
        super().__init__(*a, **kw)
        self.sample_weights, self.sources, self.feed, self.generator = sample_weights, sources, feed, None

    def _get_train_sampler(self, *a, **kw):
        if self.sample_weights is None:
            inner = super()._get_train_sampler(*a, **kw)
        else:
            if self.generator is None:
                self.generator = torch.Generator().manual_seed(int(self.args.seed))
            inner = WeightedRandomSampler(self.sample_weights, len(self.sample_weights),
                                          replacement=True, generator=self.generator)
        return CountingSampler(inner, self.sources, self.feed)

class EpochTracker(TrainerCallback):
    """Appends one row per evaluation to epoch_log.jsonl: losses, cumulative feed per source,
    and per-source validation macro-F1 / accuracy (%)."""

    def __init__(self, tokenizer, val_r, sources, feed, path, args):
        self.tokenizer, self.feed, self.path, self.a = tokenizer, feed, path, args
        self.n_by_src = Counter(sources)
        self.val_by_src = {s: [r for r in val_r if r["_src"] == s] for s in ("vcd", "conll")}

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
        losses = [h["loss"] for h in state.log_history if "loss" in h]
        row = {"epoch": round(float(state.epoch or 0), 3), "step": state.global_step,
               "eval_loss": (metrics or {}).get("eval_loss"), "train_loss_last": losses[-1] if losses else None}
        for src, n in self.n_by_src.items():
            d = self.feed["draws"][src]
            row.update({f"fed_{src}_draws": d, f"fed_{src}_distinct": len(self.feed["seen"][src]),
                        f"fed_{src}_passes": round(d / n, 3)})
        was_training = model.training
        for src, recs in self.val_by_src.items():
            if recs:
                pr = evaluate_records_seq2seq(model, self.tokenizer, recs, self.a.eval_batch_size,
                                              self.a.use_sentence, self.a.max_len)
                s = score_labels([p["gold_type"] for p in pr], [p["pred_type"] for p in pr])
                row[f"val_{src}_macro_f1"] = round(100 * float(s["macro_f1"]), 2)
                row[f"val_{src}_acc"] = round(100 * float(s["acc"]), 2)
        model.train(was_training)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


def balanced_weights(records, mix_ratio):
    """Per-example weights so that a draw is SaCTI with probability mix_ratio, VCD otherwise."""
    n_c = sum(r["_src"] == "conll" for r in records)
    n_v = len(records) - n_c
    if not n_c or not n_v:
        return None
    return [mix_ratio / n_c if r["_src"] == "conll" else (1 - mix_ratio) / n_v for r in records]


def finetune(model_name, train_r, val_r, out_dir, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    ckpt_dir = Path(out_dir) / model_name.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, warmup_ratio=0.05, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, logging_steps=100, report_to="none", dataloader_num_workers=0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # fp16 overflows ByT5's position bias and the original paper uses bf16
        seed=args.seed, data_seed=args.seed)
    sources = [r["_src"] for r in train_r]
    feed = {"draws": Counter(), "seen": defaultdict(set)}
    log_path = ckpt_dir / "epoch_log.jsonl"
    log_path.unlink(missing_ok=True)
    trainer = TrackingTrainer(
        model=model, args=targs, processing_class=tokenizer,
        train_dataset=Seq2SeqTypeDataset(train_r, tokenizer, args.max_len, args.use_sentence),
        eval_dataset=Seq2SeqTypeDataset(val_r, tokenizer, args.max_len, args.use_sentence),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        sample_weights=balanced_weights(train_r, args.mix_ratio) if args.mix_mode == "balanced" else None,
        sources=sources, feed=feed,
        callbacks=[EpochTracker(tokenizer, val_r, sources, feed, log_path, args)])
    trainer.train()

    best = ckpt_dir / "best"
    trainer.save_model(str(best))
    for d in ckpt_dir.glob("checkpoint-*"):
        shutil.rmtree(d, ignore_errors=True)
    (best / "context_config.json").write_text(
        json.dumps({"use_sentence": args.use_sentence, "max_len": args.max_len}, indent=2))
    st = trainer.state
    evals = [(e["epoch"], e["eval_loss"]) for e in st.log_history if "eval_loss" in e]
    (best / "trainer_state.json").write_text(json.dumps({
        "selection_metric": "eval_loss", "best_metric": st.best_metric,
        "best_epoch": next((ep for ep, loss in evals if loss == st.best_metric), None),
        "best_model_checkpoint": st.best_model_checkpoint, "final_epoch": st.epoch, "max_epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "n_evals": len(evals), "eval_loss_by_epoch": evals,
    }, indent=2))


def main(args):
    set_seed(args.seed)
    vcd_train, vcd_val, conll_train, conll_val = load_training_data(args)
    for records, src in ((vcd_train, "vcd"), (vcd_val, "vcd"), (conll_train, "conll"), (conll_val, "conll")):
        for r in records:
            r["_src"] = src

    if vcd_train and conll_train:
        #concat: conll repeated only when it is the smaller corpus; balanced: the sampler sets the mix.
        repeat = 1 if args.mix_mode == "balanced" else max(1, round(
            args.mix_ratio / max(1 - args.mix_ratio, 1e-6) * len(vcd_train) / len(conll_train)))
        train_r = vcd_train + conll_train * repeat
        random.shuffle(train_r)
    else:
        train_r = vcd_train or conll_train
    val_r = vcd_val + conll_val
    if not val_r:
        n_val = max(100, len(train_r) // 20)
        random.shuffle(train_r)
        val_r, train_r = train_r[:n_val], train_r[n_val:]

    for key in args.models:
        finetune(MODEL_NAMES.get(key, key), train_r, val_r, args.output, args)

    if not args.skip_baseline:
        clf = argparse.Namespace(**{k[4:]: v for k, v in vars(args).items() if k.startswith("clf_")},
                                 mix_ratio=args.mix_ratio, use_sentence=args.use_sentence, seed=args.seed)
        clf.max_len = clf.max_len or args.max_len
        train_classifier(vcd_train, vcd_val, conll_train, conll_val,
                         Path(args.output) / "classifier-baseline", clf, get_device())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--mix-mode", choices=["concat", "balanced"], default="concat",
                   help="concat: shuffled concatenation; balanced: draw SaCTI with probability --mix-ratio")
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--skip-baseline", action="store_true")
    add_classifier_args(p, "clf-")
    p.set_defaults(clf_max_len=None)
    main(p.parse_args())
