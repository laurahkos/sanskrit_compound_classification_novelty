"""run a checkpoint (typeclassifier .pt file or a byt5 directory) on an eval file; print scores and cache per-record predictions."""
import argparse
import json
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from byt5_type_eval import evaluate_records_seq2seq
from type_classifier import (detect_and_load, evaluate_records, get_device, load_model, print_scores, read_cache,
                             score_labels)

def predict(checkpoint, eval_r, batch_size=32, use_lookup=False):
    ckpt = Path(checkpoint)
    if ckpt.is_file():
        model, vocab, use_sentence = load_model(ckpt, get_device())
        lookup_path = ckpt.parent / "lookup.json"
        lookup = ({tuple(k.split("|")): v for k, v in json.loads(lookup_path.read_text(encoding="utf-8")).items()}
                  if use_lookup and lookup_path.exists() else None)
        return evaluate_records(model, eval_r, vocab, lookup, use_sentence, batch_size)
    ctx_path = ckpt / "context_config.json"
    ctx = json.loads(ctx_path.read_text()) if ctx_path.exists() else {}
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to(get_device())
    return evaluate_records_seq2seq(model, tokenizer, eval_r, batch_size,
                                    ctx.get("use_sentence", False), ctx.get("max_len", 64))


def load_predictions(source, eval_r, batch_size=32, use_lookup=False):
    """Per-record predictions from a .json cache or by running a checkpoint."""
    if str(source).endswith(".json"):
        return read_cache(source, eval_r)
    return predict(source, eval_r, batch_size, use_lookup)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--use-lookup", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()
    eval_r = detect_and_load(args.eval_data)
    per_record = predict(args.checkpoint, eval_r, args.batch_size, args.use_lookup)
    print_scores(score_labels([r["gold_type"] for r in per_record], [r["pred_type"] for r in per_record]),
                 f"{args.checkpoint} on {args.eval_data}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"checkpoint": args.checkpoint, "eval_data": args.eval_data, "per_record": per_record}, ensure_ascii=False))
