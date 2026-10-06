# Manual Training Guide — TamperResNet50

Step-by-step for running a training experiment by hand. For full
architecture/API details see `docs/tamper_resnet50_pipeline.md`.

## 1. Activate the environment

```bash
source /home/user/.venvs/ml_project/bin/activate   # or your project's venv
```

Confirm the essentials are present:
```bash
python -c "import torch, torchvision, yaml; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Make sure your dataset is laid out correctly

```
<dataset_dir>/
    authentic/   *.jpg / *.png / ...
    tampered/    *.jpg / *.png / ...
```
`data/subset` (rebalanced, from `scripts/prepare_subset.py`) or
`data/raw` (full, imbalanced) both already fit this shape.

If your source images are a mix of formats/bit-depths (e.g. some
16-bit TIFF, some JPEG, some with an alpha channel), normalize them
first so training sees uniform 8-bit RGB JPEGs:
```bash
python scripts/normalize_dataset.py \
  --authentic-dir path/to/raw/authentic \
  --tampered-dir path/to/raw/tampered \
  --output-dir path/to/normalized
```
**Put `--output-dir` on native Linux storage, not `/mnt/c/...`** if
you're on WSL2 — see `docs/tamper_resnet50_pipeline.md` §"normalize_dataset.py"
for the measured ~10-100x speed difference. This matters for training
too, since every run re-scans `dataset_dir` the same way.

## 3. Pick or copy a config

Illustrative starting-point configs live in `configs/`:
- `tamper_resnet50_baseline.yaml` — plain, unbiased.
- `tamper_resnet50_class_biased.yaml` — biased toward catching tampered images.
- `tamper_resnet50_feature_extractor.yaml` — fully-frozen backbone, for later fusion use.
- `tamper_resnet50_full_finetune*.yaml`, `tamper_resnet50_frozen_backbone_*.yaml`,
  `tamper_resnet50_layer3_layer4_combined.yaml` — the real experiment
  lineage that produced this project's actual checkpoints; see §12 of
  `docs/tamper_resnet50_pipeline.md` for what each one is and why.

To run your own variant, copy one and edit values — **never edit the
training script itself** to change an experiment:
```bash
cp configs/tamper_resnet50_baseline.yaml configs/my_experiment.yaml
```
At minimum, check/change:
- `experiment.run_name` — must be unique, or you'll overwrite a previous run's `experiments/<run_name>/` archive.
- `data.dataset_dir` — which dataset to train on.
- `model.freeze_stages`, `optim.lr_groups` — see §4 of `docs/tamper_resnet50_pipeline.md` if you change these together (every unfrozen stage + `head` must appear in `lr_groups`, or the run will fail fast with a clear error before training starts).
- `loss.use_pos_weight` / `pos_weight` and `inference.threshold` — only if you want to bias the model toward one class.
- `training.mixed_precision` — `true` (default) trains in float16 (autocast + gradient scaling) on CUDA for lower memory/faster steps; set `false` to force plain float32. Automatically falls back to float32 on CPU regardless of this setting.

## 4. Run training

```bash
python scripts/train_tamper_resnet50.py --config configs/my_experiment.yaml
```

Watch the console for, in order:
1. **Image counts** — train/val/test split sizes, confirm they look right for your dataset.
2. **Model summary** — `freeze_description()` and the parameter-shape trace; confirm the frozen/trainable split matches what you intended.
3. **Optimizer param groups** — one line per `lr_groups` key with its parameter count; confirm nothing is missing.
4. **Per-epoch lines** — train/val loss, accuracy, precision/recall/F1. Watch validation loss — that's what drives checkpointing and early stopping.
5. **"New best model saved"** — printed every time validation loss improves.
6. **Early stopping / training finished** — reports which epoch was best.
7. **Test set results** — precision/recall/F1 + confusion matrix, computed once, using `inference.threshold`.
8. **"Experiment archived to experiments/<run_name>/"** — the permanent record of this run.

A run can be interrupted (Ctrl+C) safely at any point — the last
completed epoch's checkpoint is already saved to
`paths.last_checkpoint_path`, and the best-so-far to
`paths.model_save_path`; nothing is lost except the current epoch.

## 5. Inspect the results

```
experiments/<run_name>/
    best_model.pt        # best validation-loss checkpoint
    last_model.pt         # final epoch's checkpoint
    split.json            # exact train/val/test file lists used
    config.yaml            # exact config this run used (self-contained record)
    summary.json           # results + image counts + best epoch + confusion matrix
    confusion_matrix.png
    training_history.png
```
Open `training_history.png` first for overfitting (train/val loss
diverging) and `confusion_matrix.png` for which class the model
struggles with.

## 6. Run inference on a single image

```bash
python scripts/predict_tamper_resnet50.py \
  --image path/to/some_image.jpg \
  --model experiments/<run_name>/best_model.pt \
  --mode predict
```
Use `--mode extract_features` for just the 2048-d vector, or
`--mode both` for prediction + features in one call. Add
`--threshold 0.3` to try a different decision boundary without
retraining.

## 7. Test a trained checkpoint against a different dataset

To check how a checkpoint performs on a dataset it wasn't trained on
(no split, no gradient updates — pure inference over every image in
the folder):
```bash
python scripts/evaluate_on_dataset.py \
  --model models/saved/<checkpoint>.pth \
  --dataset-dir path/to/some/dataset \
  --report-dir results/reports/<a-label-for-this-check>
```
Reports overall accuracy/precision/recall/F1 plus per-class accuracy
separately, and saves both to `results.json` + `confusion_matrix.png`
in `--report-dir`. Useful before *and* after a domain-adaptation
training run, to see exactly what changed — see §12 of
`docs/tamper_resnet50_pipeline.md` for a worked example where this
caught a catastrophic-forgetting regression that per-epoch training
logs alone wouldn't have shown (the training run's own logs only ever
see its own dataset's held-out split, never a completely separate one).

## 8. Continue training on a second dataset

To pick up a previous run's weights instead of starting from ImageNet,
set `model.init_from_checkpoint` in the new config to the earlier run's
checkpoint, point `data.dataset_dir` at the new dataset, and give it a
fresh `experiment.run_name`:
```yaml
experiment:
  run_name: "tamper_resnet50_phaseB"
data:
  dataset_dir: "data/second_dataset"
model:
  init_from_checkpoint: "experiments/tamper_resnet50_baseline/best_model.pt"
```
Then run it exactly like step 4. The console will print
`Initialized weights from checkpoint: ...` instead of starting from
ImageNet, and every checkpoint from this run records where it was
initialized from (`initialized_from` field, also in `summary.json`).
See §10 of `docs/tamper_resnet50_pipeline.md` for details and caveats
(this is continual learning, not joint training — watch for forgetting).

## 9. Common failure points

- **`lr_groups does not cover trainable stage(s) [...]`** — you changed `freeze_stages` without updating `lr_groups` to match. Every stage not frozen (plus `head`) needs an entry.
- **`Unknown stage name(s) in freeze_stages`** — typo; valid names are exactly `conv1`, `layer1`, `layer2`, `layer3`, `layer4`.
- **`Not enough images in one of the classes...`** — need at least 3 images per class for a stratified split; check `dataset_dir`.
- **Run overwrote a previous experiment** — you reused a `run_name` (and/or the shared `paths.*` from another config) without changing it.
- **Dataset scan takes minutes-to-hours instead of seconds** (WSL2 only) — `data.dataset_dir` is on `/mnt/c/...` (a Windows-drive mount). Move/normalize the dataset onto native Linux storage first (§2).
- **Good numbers on this run's own test set, bad numbers everywhere else** — that's not a bug, it's what §12 of `docs/tamper_resnet50_pipeline.md` documents happening for real: a domain-adaptation run's own held-out split only ever reflects its own training distribution. Use `scripts/evaluate_on_dataset.py` (§7) against the *other* dataset(s) to catch catastrophic forgetting before trusting a checkpoint.
