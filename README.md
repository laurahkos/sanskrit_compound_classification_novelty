## compound_classification_novelty

This repository contains code for reproducing training and evaluation for classifying two-member compounds additionally according to the lexical novelty categories scheme. It presupposes the use of two datasets with different labelling schemes, namely: 
- **VCD** (Vedic Compound Dataset, Sellmer & Hellwig 2024), labelled in accordance with the Bisetto & Scalise (2011) compound typology.
- **SaCTI** (Sandhan et al. 2022), labelled using the Sanskrit Consortium tags (Kulkarni & Kumar 2013).
SaCTI tags are mapped onto the VCD classes so that both corpora share one label space. Three models are trained on VCD, SaCTI, or both, with and without the surrounding sentence. Evaluation is stratified by how novel each test compound is relative to the training data.

## Contents

| File | Role |
|---|---|
| `type_classifier.py` | Label space and SaCTI→VCD mapping, data loading, metrics, and the small character-level Transformer (`TypeClassifier`). |
| `preprocess_vcd.py` | VCD CSV → `type/{train,val,test}.jsonl`. |
| `dedup_resplit_sacti.py` | SaCTI original fine-grained CoNLL split files → transliterated, deduplicated, re-split `{train,dev,test}.jsonl`. |
| `byt5_type_eval.py` | Fine-tunes `byt5-base` and `byt5-sanskrit`. |
| `cache_predictions.py` | Runs a checkpoint on a test file and saves per-item predictions as json. |
| `stratify_eval.py` | Scores one prediction cache by novelty category. |
| `seed_compare.py` | Compares two systems across seeds with a two-level (seed × item) bootstrap. |
| `bootstrap_compare.py` | The same comparison for two single runs (item bootstrap only). |
| `karaka_axis_t3.py` | SaCTI accuracy grouped by kāraka status of the tag, with and without T3. |
| `context_ambiguity.py` | Effect of sentence context on compound strings that occur with more than one type. |
| `summarise_results.py` | Collects everything into Markdown tables. |
| `job_train.slurm` | One training run: one condition x one seed. |
| `job_posthoc.slurm` | All analyses over the cached predictions put into a summary. |

## Requirements

Python 3.10+ with `torch`, `transformers`, `scikit-learn`, `pandas` and `numpy`. ByT5 training uses bf16. Slurm scripts were written to be run on the institution cluster and assume a single A40 GPU (partition `gpuA40GB`) and `module load libs/cuda`.

## Labels

Five classes (`TYPE_LABELS`): `subord_endo`, `subord_exo`, `attrib_endo`, `attrib_exo`, `coord_endo`. In the VCD, the class is `cpdType` + `endoExo`. Coordinate compounds are always `coord_endo`, and rows whose combination falls outside the five classes are dropped. SaCTI tags are mapped by `CONLL_FINE_TO_VCD` in the `type_classifier.py` script in the following way:

| VCD class | SaCTI tags |
|---|---|
| `subord_endo` | T6, T4, T7, U |
| `attrib_endo` | K1, Km, K6, K7, Tn, Tp, T3 |
| `subord_exo` | Bs6, BvS |
| `coord_endo` | Di |
| (dropped) | A1 (avyayībhāva) |

SaCTI has no `attrib_exo`. For cross-corpus comparisons, pass `--merge-exo` to score the two exocentric classes as a single `exo` class. Without it, a SaCTI-trained model cannot exceed 80 five-class macro-F1 on the VCD.

## Data preparation
The raw data is not in this repository, it belongs to their rightful owners, namely Sellmer & Hellwig (2024) and Sandhan et al. (2022).
```bash
python preprocess_vcd.py --input VCD-final.csv --output data/vcd
```
Script Reads the tab-separated `VCD-final.csv` and writes `data/vcd/type/{train,val,test}.jsonl`, an 80/10/10 split stratified by class.
Each record keeps `compound`, `lemma1`, `lemma2`, `full_type` and `sentence`. `intLabel`, `pos1` and `pos2` are kept as `_*_raw` for provenance only and never reach the models.
**SaCTI**
```bash
python dedup_resplit_sacti.py --conll train.conll dev.conll test.conll --output-dir data/sacti_dedup_context
```
File reads each sentence block from the CoNLL files. The compound is the root token (head `0`), and its tag is the root's dependency label. It transliterates Devanagari to IAST. This also repairs UTF-8 mojibake, drops nukta, and writes SaCTI's `:` visarga as `ḥ`. It then pools all three files and deduplicates on (compound, sentence). If identical (compound, sentence) pairs carry different tags, they are dropped and written to `dropped_ambiguous.jsonl`, then re-splits 80/10/10, stratified by mapped class.

## Training conditions

| Condition | Training data |
|---|---|
| `cond1` | VCD only |
| `cond2` | SaCTI only |
| `cond3` | VCD + SaCTI, concatenated |
| `cond3B` | VCD + SaCTI, balanced sampling |
| `…S` (e.g. `cond3BS`) | same as the base condition, but with sentential context appended |

Settings that change with the context flag:
| Setting | Without context | With context (`…S`) |
|---|---|---|
| `max_len` (bytes/characters) | 64 | 256 |
| ByT5 micro-batch | 32 | 16 |
| Gradient accumulation | 1 | 2 |

## Models

**ByT5** (`byt5_type_eval.py`): `google/byt5-base` and `buddhist-nlp/byt5-sanskrit`.

Fine-tuned as seq2seq, with the class name as the target string. Iy defaults to 15 epochs, learning rate 5e-5, warm-up ratio 0.05. It is evaluated every epoch; the checkpoint with the lowest validation loss is kept. Prediction uses **rank classification**, not free generation. Each evaluation appends a row to `epoch_log.jsonl` with the training and validation losses, per-corpus validation macro-F1 and accuracy, and how many examples from each corpus have been fed so far (draws, distinct items, passes).

**TypeClassifier** (`type_classifier.py`): a character-level Transformer encoder trained from scratch (about 4.8M parameters).

Architecture: 6 layers, hidden size 256, 8 heads, FFN 1024, sinusoidal positions, mean pooling, and a 2-layer classification head.
The character vocabulary is built from that condition's own data. Loss is cross-entropy with label smoothing 0.05 and square-root-inverse class weights, the optimiser is AdamW at 1e-4, linear warm-up, and learning-rate reduction on plateau. Up to 40 epochs, with early stopping on validation macro-F1 (patience 8). Classifier hyperparameters are set with `--clf-*` flags in `byt5_type_eval.py`, or without the prefix when running `type_classifier.py` directly. The classifier also writes `lookup.json`, a VCD lemma-pair → type table as it was checked whether lookup would benefit the classifier. It is applied only with `--use-lookup`, which the pipeline does not pass, so the reported scores are purely from the neural network.

## Running
**One condition and seed:**
```bash
sbatch job_train.slurm cond3BS 42
```
The above does the following:
1. Trains both ByT5 models and `TypeClassifier`, with outputs under `runs/<cond>/seed<seed>/`.
2. Caches per-item predictions on both test sets to `cache/<cond>/seed<seed>/<tag>_{vcd,sacti}.json`, along with each model's `trainer_state.json` and `epoch_log.jsonl`.
3. Runs the kāraka analysis from those caches, writing to `karaka/<cond>/seed<seed>/`.
4. Deletes the checkpoints of every model whose caches were written successfully.
Model tags are `google_byt5-base`, `buddhist-nlp_byt5-sanskrit` and `classifier`. `job_info.txt` in the run directory records the date, GPU, settings and the SaCTI mapping in force. The paper uses seeds 0, 1, 2 and 42 for all eight conditions:
```bash
for c in cond1 cond1S cond2 cond2S cond3 cond3S cond3B cond3BS; do
  for s in 0 1 2 42; do sbatch job_train.slurm $c $s; done
done
```
The project directory defaults to `~/iscls_code`. Override it, or the output roots, with `PROJECT_DIR`, `RUNS_ROOT`, `CACHE_ROOT` and `KARAKA_ROOT`.

## Analyses

**Novelty strata** (`stratify_eval.py`). Each test compound is placed in a category relative to the training data:

| Category | Meaning |
|---|---|
| `1.1` | Compound string seen |
| `1.2a` | Member pair seen, in a different surface form |
| `1.2b` | Both members seen, never together |
| `1.2` | `1.2a` ∪ `1.2b` |
| `2` | One member unseen |
| `0` | Both members unseen |

How members are identified:
- **VCD:** by `lemma1`/`lemma2`.
- **SaCTI:** by the surface segments either side of the compound's hyphen, since it has no lemmas. Most new forms of seen pairs therefore land in `1.2b` rather than `1.2a`.
- 
 What counts as "seen" depends on the condition:
- `cond1`: the VCD training split.
- `cond2`: the SaCTI training split.
- Joint conditions: both training splits.

**Significance** (`seed_compare.py`). Differences between two systems are mean-over-seeds differences in macro-F1 and accuracy. The 95% interval comes from resampling seeds and test items jointly, and the p-value is two-sided. Results are reported overall and per novelty category.

**Kāraka groups** (`karaka_axis_t3.py`). Group membership is by SaCTI tag:

| Group | Tags |
|---|---|
| Explicit kāraka | T3, T4, T7 |
| Explicit, without T3 | T4, T7 |
| Covert | U |
| Non-kāraka | everything else |

**Context and ambiguity** (`context_ambiguity.py`). A compound string counts as type-ambiguous if it carries more than one class anywhere in the corpus, across all splits. The script compares context against no context separately on ambiguous and unambiguous test items.

**Summary tables** (`summarise_results.py`):

| File | Contents |
|---|---|
| `main.md` | Overall scores, mean ± sd over seeds. |
| `novelty.md` | Macro-F1 per novelty category. |
| `tags.md` | SaCTI accuracy per tag. |
| `confusions.md` | Prediction distribution for T3, T7, U, K1 and K6. |
| `t3_split.md` | T3 split by licensing sūtra, A 2.1.31 vs. 2.1.32. This split is heuristic (second-member list and participle endings) and should be checked by hand. |
| `errors.md` | Sample misclassifications. |

## References

- Bisetto, A. & Scalise, S. (2011). The classification of compounds.
- Kulkarni, A. & Kumar, A. (2013). Clues from Aṣṭādhyāyī for compound type identification.
- Sandhan, J. et al. (2022). A novel multi-task learning approach for context-sensitive compound type identification in Sanskrit. arXiv:2208.10310.
- Sellmer, S. & Hellwig, O. (2024). The Vedic Compound Dataset.
- Xue, L. et al. (2022). ByT5: Towards a token-free future with pre-trained byte-to-byte models.
- Nehrdich, S. et al. (2024). One model is all you need: ByT5-Sanskrit.
