# TamperResNet50 Pipeline — Architecture & Reference

Documentation for the config-driven ResNet50 authentic-vs-tampered
classifier: `configs/tamper_resnet50_*.yaml`,
`src/models/tamper_resnet50.py`, `scripts/train_tamper_resnet50.py`,
`scripts/predict_tamper_resnet50.py`, `scripts/normalize_dataset.py`,
`scripts/evaluate_on_dataset.py`, and the shared data/util modules they
reuse from the existing baseline pipeline.

**If you just want the current best checkpoint and its path, skip to
§12.**

This pipeline is additive: `scripts/train.py`, `scripts/evaluate.py`,
`scripts/predict.py`, and `src/models/resnet_model.py` (the
baseline/deep/resnet50 comparison pipeline) are untouched and keep
working exactly as before. `src/data/preprocessing.py` and
`src/data/dataloader.py` gained new optional parameters (all default to
the old behavior) so both pipelines share one data layer.

---

## 1. Why a separate model class

`src/models/resnet_model.py::ResNetCNN` (existing) pools ResNet50's
2048-d output down to a 128-d embedding before its output head, and
only supports freezing a *count* of leading stages
(`num_frozen_layers: int`). That hides the standard 2048-d penultimate
feature vector this task needs for multi-agent fusion, and the
count-based freeze API is exactly the ambiguity the brief called out
("`num_frozen_layers: 2` wasn't clear which layers that meant").

`src/models/tamper_resnet50.py::TamperResNet50` (new) is architecturally
closer to plain ResNet50: stem → layer1-4 → global-avg-pool → **2048-d
feature vector** → dropout → single-logit FC. Every design lever the
brief asked for (explicit stage list, independent BN freezing,
differential LR, feature/prediction call modes) is a constructor
argument or method on this class, not something the training script
has to special-case.

---

## 2. End-to-end flow

```
configs/tamper_resnet50_*.yaml
        │  (one YAML = one experiment variant; training script never changes)
        ▼
scripts/train_tamper_resnet50.py
        │
        ├── src/utils/helpers.py            set_seed(), get_device()
        │
        ├── src/data/dataloader.py          create_dataloaders()
        │       ├── src/data/dataset.py         build_dataset_index(), ImagePathDataset
        │       └── src/data/preprocessing.py    get_train_transforms(), get_eval_transforms()
        │
        ├── src/models/tamper_resnet50.py   TamperResNet50(..., pretrained=init_from_checkpoint is None)
        │       ├── [optional] .load_state_dict(prior checkpoint)  -> continue training (§10)
        │       ├── .get_param_groups(lr_groups)   -> differential-LR optimizer groups
        │       └── .forward() / .train()          -> stage & BN freezing applied every step
        │
        ├── torch.optim.Adam(param_groups, weight_decay=...)
        ├── torch.nn.BCEWithLogitsLoss(pos_weight=...)   (optional)
        │
        ├── training loop (epochs_cap, EarlyStopping on val loss)
        │       ├── checkpoint EVERY epoch  -> paths.last_checkpoint_path
        │       └── checkpoint on new best  -> paths.model_save_path
        │
        ├── src/utils/metrics.py            compute_metrics(), compute_confusion(), print_metrics()
        ├── src/utils/visualize.py          plot_confusion_matrix(), plot_training_history()
        │
        └── archive_experiment()            experiments/<run_name>/  (checkpoints + plots + config.yaml + summary.json)

scripts/predict_tamper_resnet50.py
        │
        ├── src/data/preprocessing.py       get_eval_transforms()
        └── src/models/tamper_resnet50.py   TamperResNet50.predict() / .extract_features() / .predict_with_features()
```

---

## 3. File-by-file reference

### `configs/tamper_resnet50_baseline.yaml`
### `configs/tamper_resnet50_class_biased.yaml`
### `configs/tamper_resnet50_feature_extractor.yaml`

**Role:** the *only* place experiment design choices live. The training
script reads these and never branches on "which variant is this" — the
three example files just set different values for the same schema.

Top-level schema (all keys required):

| Section | Key | Meaning |
|---|---|---|
| `experiment` | `run_name` | Folder name under `experiments/` for this run's permanent record. |
| | `seed` | Passed to `set_seed()` and to the train/val/test split. |
| `data` | `dataset_dir` | Must contain `authentic/` and `tampered/` subfolders. |
| | `img_size` | `LetterboxResize` target (longest side → this, then padded to a square). |
| | `batch_size`, `val_ratio`, `test_ratio`, `num_workers` | Passed straight to `create_dataloaders()`. |
| | `augment` | Master switch for `hflip`/`random_crop`/`color_jitter` together. |
| | `hflip`, `random_crop`, `color_jitter` | Individual augmentation toggles (only apply when `augment: true`). |
| | `jpeg_recompression`, `jpeg_quality_range` | **Independent** of `augment` — see §5. |
| `model` | `init_from_checkpoint` | `null` for a normal ImageNet-pretrained start, or a previous run's checkpoint path to continue training from it (e.g. on a second dataset) — see §10. |
| | `freeze_stages` | List drawn from `["conv1","layer1","layer2","layer3","layer4"]`. |
| | `freeze_bn` | Freezes BatchNorm running stats network-wide, independent of `freeze_stages`. |
| | `dropout` | Applied to the pooled 2048-d vector before the final FC layer. |
| `optim` | `weight_decay` | Adam weight decay (one value, applied to every param group). |
| | `lr_groups` | `{group_name: lr}` — see §4. |
| `loss` | `use_pos_weight`, `pos_weight` | Biases **training** — see §6. |
| `inference` | `threshold` | Biases **inference** — see §6. |
| `training` | `epochs_cap`, `early_stopping_patience` | Passed to the training loop / `EarlyStopping`. |
| | `mixed_precision` | `true` = float16 autocast + gradient scaling instead of float32 (CUDA only, auto-falls-back to float32 on CPU) — see §11. |
| `paths` | `model_save_path`, `last_checkpoint_path`, `split_save_path`, `plots_dir`, `experiments_dir` | Every output location — nothing is hardcoded in the script. |

To create a fourth variant: copy one of these files, change values, run
with `--config path/to/new.yaml`. No Python changes needed.

---

### `src/models/tamper_resnet50.py`

**Role:** the model. Owns freezing, differential LR grouping, forward
pass, and all three inference call modes.

Module-level constants:
- `STAGE_NAMES = ("conv1", "layer1", "layer2", "layer3", "layer4")`
- `GROUP_NAMES = STAGE_NAMES + ("head",)`

#### `class TamperResNet50(nn.Module)`

**`__init__(freeze_stages=None, freeze_bn=False, dropout=0.5, pretrained=True)`**
Builds ResNet50 (`torchvision.models.resnet50`, `ResNet50_Weights.IMAGENET1K_V2`
when `pretrained=True`), splits it into named sub-modules
(`self.stem`, `self.layer1..4`, `self.global_pool`, `self.fc`), applies
`freeze_stages` immediately via `requires_grad`. Raises `ValueError` on
an unknown stage name.

**`train(mode: bool = True) -> self`** *(overridden)*
Called every epoch by `model.train()`/`model.eval()`. After the normal
`nn.Module.train()`, re-forces:
1. any stage in `freeze_stages` back into `.eval()` (its weights don't
   move, so its BN shouldn't drift either), and
2. if `freeze_bn=True`, **every** `BatchNorm{1,2,3}d` in the backbone
   into `.eval()`, even inside stages that are otherwise training.

This is what makes `freeze_bn` and `freeze_stages` genuinely
independent — see §7 for the truth table.

**`freeze_description() -> str`**
Human-readable one-liner, e.g. `"frozen stages: [conv1, layer1] |
freeze_bn: False"`. Printed at the start of every training run.

**`get_param_groups(lr_groups: Dict[str, float]) -> List[dict]`**
Turns a config dict into `torch.optim`-ready param groups. Each key is
one or more `GROUP_NAMES` joined by `"_"` (e.g. `"conv1_layer1"`). Every
stage that is *not* in `freeze_stages` (plus `"head"`) must be covered
by exactly one key, or this raises `ValueError` naming the missing
stage(s) — a coverage gap here would silently train part of the network
with no LR assigned, so it's a hard error instead of a fallback. Also
raises if a stage appears in two keys. Returns
`[{"params": [...], "lr": ..., "name": key}, ...]`, ready for
`torch.optim.Adam(param_groups, weight_decay=...)`.

**`forward(x: Tensor, return_features: bool = False) -> Tensor | (Tensor, Tensor)`**
Raw logits, shape `(batch, 1)`, **no sigmoid** (matches
`BCEWithLogitsLoss`, which wants raw logits). With
`return_features=True`, also returns the `(batch, 2048)` pooled feature
vector from the same forward pass (no second backbone pass needed).

**`extract_features(x: Tensor) -> Tensor`**
`(batch, 2048)` feature vector — post global-average-pool, pre-FC.
Forces `self.eval()` and runs under `torch.no_grad()` internally (this
is an inference-only method). Independent of the FC layer: a caller
never needs to touch classification-head weights to get features, which
is what makes a trained checkpoint reusable as a feature source
("Agent A/B") in a later fusion model without re-architecting anything.

**`predict(x: Tensor, threshold: float = 0.5) -> (labels, confidences, logits)`**
Inference-only. `labels`: `LongTensor (batch,)`, 1=tampered/0=authentic.
`confidences`: `FloatTensor (batch,)`, **always P(tampered)** (sigmoid
of the logit) — not "probability of the predicted class" — so it's
comparable across calls regardless of which label won. `logits`: raw
pre-sigmoid scores, kept for debugging/downstream use. `threshold` only
affects `labels`; it never touches model weights (§6).

**`predict_with_features(x: Tensor, threshold: float = 0.5) -> (labels, confidences, logits, features)`**
Same as `predict()` plus the `(batch, 2048)` feature vector, computed in
one forward pass via `forward(x, return_features=True)`. This is the
single call an Agent A/B fusion setup needs.

**`get_last_conv_layer() -> nn.Conv2d`**
Last `Conv2d` in `layer4`'s final block — reserved for Grad-CAM later
(not used by this pipeline yet).

#### `print_shape_trace(model, img_size=224, in_channels=3) -> None`
Module-level helper (not a class method). Pushes a dummy zero tensor
through the model, prints the shape after pooling and after the FC
layer, plus a trainable/total parameter count and
`model.freeze_description()`. Called once at the start of training for
a sanity check on what's actually frozen.

---

### `scripts/train_tamper_resnet50.py`

**Role:** the training entry point. Reads one YAML config, wires
together the data/model/loss/optimizer, runs the training loop, and
archives the run. Contains **no hardcoded experiment values** — every
number/flag referenced comes from `cfg[...]`.

CLI:
```
python scripts/train_tamper_resnet50.py --config configs/tamper_resnet50_baseline.yaml
```
`--config` defaults to `configs/tamper_resnet50_baseline.yaml` if omitted.

Key functions:

- **`load_config(path) -> dict`** — thin `yaml.safe_load` wrapper.
- **`train_one_epoch(model, loader, criterion, optimizer, device) -> (loss, acc)`**
  — one pass over the train loader, gradient updates included. Uses a
  fixed 0.5 threshold for the accuracy metric it reports (monitoring
  only — see §6, this is not `cfg["inference"]["threshold"]`).
- **`evaluate_epoch(model, loader, criterion, device) -> (loss, acc, precision, recall, f1)`**
  — `@torch.no_grad()`, called on the val loader every epoch, used for
  early stopping and the per-epoch log line. Also fixed at 0.5.
- **`run_test_evaluation(model, loader, device, threshold) -> (y_true, y_pred)`**
  — `@torch.no_grad()`, called once at the end on the held-out test set
  with `cfg["inference"]["threshold"]` — the one place besides
  `predict()`/`predict_with_features()` where that threshold actually
  changes a reported number.
- **`archive_experiment(cfg, run_dir, checkpoint, test_metrics, confusion, stopped_at_epoch, image_counts)`**
  — copies `best_model.pt`/`last_model.pt`/`split.json`/plots into
  `experiments/<run_name>/`, and writes `config.yaml` (the exact config
  used) + `summary.json` (results, image counts, which epoch was best).
- **`main()`** — orchestrates everything above; see §2's flow diagram.

Per-epoch checkpoint contents (`build_checkpoint()` closure inside
`main()`), saved to both `last_checkpoint_path` (every epoch) and
`model_save_path` (only on a new best val loss):
```python
{
    "model_state_dict": ...,
    "model_class": "TamperResNet50",
    "model_init_kwargs": {"freeze_stages": ..., "freeze_bn": ..., "dropout": ...},
    "img_size": ...,
    "epoch": ...,
    "val_loss": ..., "val_acc": ...,
    "class_names": ["authentic", "tampered"],
    "inference_threshold": ...,       # cfg["inference"]["threshold"], carried into the checkpoint
    "image_counts": {"train": N, "val": N, "test": N, "total": N},
    "run_name": ...,
}
```
`model_init_kwargs` is what lets `scripts/predict_tamper_resnet50.py`
reconstruct the exact same architecture from the checkpoint alone,
without needing the original YAML file.

---

### `scripts/predict_tamper_resnet50.py`

**Role:** demonstrates/exercises the three checkpoint call modes from
the CLI — this is also the reference implementation for how a future
fusion script should load and call a trained checkpoint.

CLI:
```
python scripts/predict_tamper_resnet50.py \
  --image path/to/image.jpg \
  --model models/saved/tamper_resnet50_baseline_best.pth \
  --mode {predict, extract_features, both}  \
  [--threshold 0.4]
```

- **`load_model(checkpoint_path, device) -> (model, checkpoint)`** —
  `torch.load` the checkpoint, rebuild `TamperResNet50(**checkpoint["model_init_kwargs"], pretrained=False)`
  (skips re-downloading ImageNet weights since `load_state_dict` will
  overwrite them anyway), loads weights, sets `model.eval()`.
- **`load_tensor(image_path, img_size, device) -> Tensor`** — reuses
  `get_eval_transforms()` from `src/data/preprocessing.py` (the exact
  deterministic pipeline used for val/test, so predictions match what
  was measured during training) and adds a batch dimension.
- **`main()`** — parses args, loads model + image, dispatches to
  `model.predict()` / `model.extract_features()` / `model.predict_with_features()`
  depending on `--mode`, prints the result. `--threshold` overrides the
  checkpoint's saved `inference_threshold` for this call only — no
  retraining, matches the design in §6.

---

### `scripts/normalize_dataset.py`

**Role:** one-time dataset preparation, run *before* training, not part
of the training script itself. Converts an arbitrary source folder
(mixed formats/bit-depths/modes, e.g. a mix of JPEG and 16-bit TIFF) to
uniform 8-bit RGB JPEGs at a fixed quality, mirroring `authentic/` and
`tampered/` subfolders into an output directory. Uses a
`multiprocessing.Pool` (one worker per core by default) since a
real-sized dataset (hundreds of thousands of images) makes this
single-threaded otherwise.

CLI:
```
python scripts/normalize_dataset.py \
  --authentic-dir path/to/raw/authentic \
  --tampered-dir path/to/raw/tampered \
  --output-dir path/to/normalized \
  --quality 90
```

**Critical caveat learned the hard way**: the output directory must be
on a **native Linux filesystem**, not a Windows-drive mount under WSL2
(`/mnt/c/...`). Reading/writing hundreds of thousands of individual
small files through WSL2's DrvFs translation layer measured
**~40-450 files/s** depending on parallelism, vs. **~1,300-4,500
files/s** on native ext4 (e.g. `/home/<user>/...`) — the difference
between a dataset scan/copy taking a minute and taking hours. This
matters again every time `scripts/train_tamper_resnet50.py` scans
`data.dataset_dir` (via `build_dataset_index`, itself single-threaded),
so **training datasets should live on native storage**, not `/mnt/c`.

---

### `scripts/evaluate_on_dataset.py`

**Role:** pure inference/testing of an already-trained checkpoint
against an arbitrary dataset directory — no training, no split, no
gradient computation. Distinct from `run_test_evaluation()` inside
`train_tamper_resnet50.py` (which only ever sees the held-out test
slice of whatever dataset a run trained on): this script evaluates a
checkpoint against a *different* dataset entirely (e.g. "how does the
308K-trained model do on a completely separate dataset it never saw at
all, train/val/test included").

CLI:
```
python scripts/evaluate_on_dataset.py \
  --model models/saved/<checkpoint>.pth \
  --dataset-dir path/to/some/dataset \
  --report-dir results/reports/<label> \
  --batch-size 96
```

Guarantees the checkpoint is frozen: `model.eval()` + every forward
pass wrapped in `torch.no_grad()`, so no amount of re-running this
script can ever change a checkpoint's weights. Reports overall
accuracy/precision/recall/F1 (reusing `src/utils/metrics.py`) **and**
per-class accuracy separately (`per_class_accuracy` in the saved
`results.json`) — important when, e.g., one class in the target
dataset overlaps with what the checkpoint was trained on and the other
doesn't, so a single blended accuracy number would be misleading (see
§12's authentic-vs-tampered overlap case).

---

## 4. Differential learning rates — worked example

Given (from `configs/tamper_resnet50_baseline.yaml`):
```yaml
model:
  freeze_stages: ["conv1", "layer1"]
optim:
  lr_groups:
    layer2: 0.00003
    layer3: 0.0001
    layer4: 0.0003
    head: 0.001
```

`model.get_param_groups(lr_groups)`:
1. Splits each key on `_`: `"layer2"→["layer2"]`, ..., `"head"→["head"]`.
2. Builds `stage_to_key = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4", "head": "head"}`.
3. Checks every stage *not* in `freeze_stages` (`layer2, layer3, layer4`)
   plus `head` is present → yes, no `ValueError`.
4. For each key, collects `p for p in <module>.parameters() if p.requires_grad`
   (frozen `conv1`/`layer1` params are `requires_grad=False`, so even if
   they were mistakenly referenced they'd contribute nothing).
5. Returns 4 param groups, handed straight to
   `Adam(param_groups, weight_decay=cfg["optim"]["weight_decay"])`.

A combined key like `"conv1_layer1": 1e-5` (as in the brief's own
example) works the same way when `conv1`/`layer1` are *not* frozen —
both stage names get mapped to that one key and share its LR.

---

## 5. Augmentation pipeline

`src/data/preprocessing.py::get_train_transforms(img_size, augment, hflip, random_crop, color_jitter, jpeg_recompression, jpeg_quality_range, crop_pad=16)`:

```
LetterboxResize(img_size)              # always applied, longest side -> img_size, padded to square
  │
  ├─ if augment and random_crop:  Pad(crop_pad, reflect) -> RandomCrop(img_size)
  ├─ if augment and hflip:        RandomHorizontalFlip(p=0.5)
  ├─ if augment and color_jitter: ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.02)
  │
  ├─ if jpeg_recompression:       RandomJPEGRecompression(quality_range, p=0.5)   # INDEPENDENT of `augment`
  │
  ToTensor()
  Normalize(NORM_MEAN, NORM_STD)       # ImageNet stats, always applied
```

`augment: false` disables the flip/crop/jitter group in one switch
(for a controlled experiment that isolates some other variable).
`jpeg_recompression` has its own switch on purpose: it changes the
JPEG-artifact signal that ELA-based explainability (`src/features/ela.py`)
reads, so a run may need it off even when other augmentation is on, or
on even when other augmentation is off.

`get_eval_transforms(img_size)` (unchanged) is always
`LetterboxResize → ToTensor → Normalize` — deterministic, used for
val/test and every single-image prediction, so a reported metric never
depends on random augmentation draws.

`src/data/dataloader.py::create_dataloaders(...)` forwards
`augment/hflip/random_crop/color_jitter/jpeg_recompression/jpeg_quality_range`
straight through to `get_train_transforms()` for the train split only;
val/test always get `get_eval_transforms()` regardless of these flags.

---

## 6. The two independent bias levers

| | `loss.use_pos_weight` / `loss.pos_weight` | `inference.threshold` |
|---|---|---|
| **What it touches** | The loss function (`nn.BCEWithLogitsLoss(pos_weight=...)`) | The decision boundary applied to `sigmoid(logit)` |
| **When it acts** | During training — changes gradients, so it changes *what the model learns* | After training — changes nothing about the weights |
| **Where it's read** | `scripts/train_tamper_resnet50.py::main()`, building `criterion` | `run_test_evaluation()` (final test report), and `TamperResNet50.predict()`/`.predict_with_features()` (any later inference call) |
| **Can it change without retraining?** | No — a new `pos_weight` means retraining | Yes — pass a different `threshold` to `predict()`/`predict_with_features()`, or override `--threshold` on `scripts/predict_tamper_resnet50.py`, any time |

They can be used alone, together, or neither — `configs/tamper_resnet50_class_biased.yaml`
uses both (`pos_weight: 1.4` and `threshold: 0.4`) to bias twice toward
catching tampered images; `configs/tamper_resnet50_baseline.yaml` uses
neither.

Per-epoch train/val accuracy/precision/recall during training
(`train_one_epoch`, `evaluate_epoch`) always use a fixed 0.5 boundary,
deliberately independent of `inference.threshold` — that threshold is a
post-hoc deployment knob; letting it influence early-stopping/
checkpoint-selection would make model-selection decisions depend on a
number chosen for a different purpose.

---

## 7. Freezing — stage weights vs. BatchNorm stats

`freeze_stages` (which conv weights can move) and `freeze_bn` (whether
BN running stats can drift) are independent flags. Truth table for a
given stage:

| `freeze_stages` includes it? | `freeze_bn`? | Conv weights | BN running stats |
|---|---|---|---|
| No | `false` | trainable | update normally (default) |
| No | `true` | trainable | **frozen** (`.eval()`) even though weights train |
| Yes | `false` | **frozen** | frozen (a frozen stage is always put in `.eval()` too — see `train()` override) |
| Yes | `true` | **frozen** | frozen (redundant with the row above, harmless) |

The practically useful row is (No, `true`): fine-tune a stage's
convolutions while keeping its BatchNorm statistics pinned to the
ImageNet-pretrained values — useful when `batch_size` is small enough
that fresh BN statistics from this dataset would be noisy.
`configs/tamper_resnet50_feature_extractor.yaml` uses (Yes, `true`) for
every stage, so the 2048-d `extract_features()` output is entirely
determined by the frozen, pretrained backbone.

---

## 8. Reused components (unchanged behavior)

| File | Used for |
|---|---|
| `src/data/dataset.py` | `CLASS_TO_LABEL`/`LABEL_TO_CLASS` (`authentic=0, tampered=1`), `build_dataset_index()` (scans `<dataset_dir>/authentic`, `<dataset_dir>/tampered`, verifies each file with `PIL.Image.verify()`), `ImagePathDataset` (`(path, label)` list → `(tensor, label_tensor)` on `__getitem__`). |
| `src/utils/helpers.py` | `set_seed(seed)` (Python/NumPy/Torch/CUDA + `PYTHONHASHSEED`), `get_device()` (CUDA if available, else CPU, prints which), `EarlyStopping(patience, min_delta=0.0)` (`.step(val_loss) -> bool` is-this-a-new-best, `.should_stop`, `.best_loss`). |
| `src/utils/metrics.py` | `compute_metrics(y_true, y_pred) -> {accuracy, precision, recall, f1}` (positive class fixed to `tampered=1`), `compute_confusion(y_true, y_pred)` (`[authentic, tampered] x [authentic, tampered]`), `print_metrics(metrics)`. |
| `src/utils/visualize.py` | `plot_training_history(history, save_path)` (loss/accuracy curves), `plot_confusion_matrix(cm, class_names, save_path)`. |

None of these files were changed. `src/data/preprocessing.py` and
`src/data/dataloader.py` gained new *optional* parameters (§5) — every
existing call site (`scripts/train.py`, `scripts/evaluate.py`,
`scripts/predict.py`) still works unmodified because the new
parameters default to the pipeline's original behavior.

---

## 9. Output artifacts per run

Given `run_name: "tamper_resnet50_baseline"`:

```
models/saved/tamper_resnet50_baseline_best.pth        # best-val-loss checkpoint (overwritten by next run using this path)
models/checkpoints/tamper_resnet50_baseline_last.pth  # last-epoch checkpoint
data/processed/split_tamper_resnet50_baseline.json    # exact train/val/test file lists used
results/plots/tamper_resnet50_baseline/
    confusion_matrix.png
    training_history.png

experiments/tamper_resnet50_baseline/                 # permanent record, never overwritten by a later run
    best_model.pt
    last_model.pt
    split.json
    confusion_matrix.png
    training_history.png
    config.yaml            # exact config this run used
    summary.json           # results + image_counts + best_checkpoint_epoch + test_metrics + confusion_matrix
```

`models/saved/`, `models/checkpoints/`, and `experiments/` are
git-ignored (see `.gitignore`) — only `experiments/<run_name>/` is
meant to be a durable, shareable record of one specific run's exact
config and results.

---

## 10. Continuing training on a second dataset

`model.init_from_checkpoint` (default `null`) makes this a config
value, not a code change:

```yaml
experiment:
  run_name: "tamper_resnet50_phaseB"   # must differ from phase A's run_name
data:
  dataset_dir: "data/second_dataset"    # the new dataset
model:
  init_from_checkpoint: "experiments/tamper_resnet50_baseline/best_model.pt"
```

What happens in `scripts/train_tamper_resnet50.py::main()` when this is set:
1. `TamperResNet50(...)` is constructed with `pretrained=False` — ImageNet
   weights are skipped entirely, since they'd be overwritten in the next
   line anyway.
2. `model.load_state_dict(torch.load(init_from_checkpoint)["model_state_dict"])`
   loads the previous run's weights. This always succeeds regardless of
   what `freeze_stages`/`freeze_bn`/`dropout` either run used — those are
   behavioral flags, not architecture changes, so the checkpoint's
   parameter names are identical either way.
3. Everything else — `freeze_stages`, `lr_groups`, augmentation, loss,
   epochs, early stopping, the train/val/test split — comes from
   *this* run's own config, not the previous one. So a second phase can
   freeze more (or less) of the backbone, or use different learning
   rates, than the first phase did.
4. Every checkpoint this run saves records `"initialized_from": <path>`
   (`null` for a from-ImageNet run), and it's also in
   `experiments/<run_name>/summary.json` under `config.model.init_from_checkpoint`
   — so a later run's provenance (which checkpoint, which prior dataset)
   is always traceable from its own archived config.

Caveats: this is real continual learning, not joint training on both
datasets — catastrophic forgetting on dataset A is possible if dataset
B differs a lot in distribution or class balance. Freezing more stages
for the second phase, and/or using smaller `lr_groups` values than a
from-scratch run, are the usual mitigations. There is currently no
automatic check that the two datasets don't overlap — if that matters,
diff the two runs' `split.json` files.

---

## 11. Mixed precision (float16) training

`training.mixed_precision: true` (default in all three example configs)
switches training from plain float32 to **automatic mixed precision**:
the forward pass and loss are computed in float16 (`torch.autocast`),
while a `torch.amp.GradScaler` rescales the loss before `.backward()`
so float16 gradients don't underflow to zero, then unscales them again
before the optimizer step. Model weights themselves stay float32 the
whole time — this is not `model.half()` (running BatchNorm and
accumulating gradients purely in float16 is numerically unstable);
autocast picks float16 only for the ops that tolerate it (mostly the
conv/linear matmuls) and keeps float32 where it matters.

Implementation (`scripts/train_tamper_resnet50.py`):
- **`resolve_amp(mixed_precision, device) -> bool`** — the config flag
  only takes effect on CUDA; on CPU it's silently disabled (a note is
  printed) since float16 autocast isn't a CPU win. `main()` calls this
  once and threads the resulting `use_amp` bool through
  `train_one_epoch`, `evaluate_epoch`, and `run_test_evaluation`.
- **`train_one_epoch(..., use_amp, scaler)`** — forward pass + loss
  inside `torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)`,
  then `scaler.scale(loss).backward()`, `scaler.step(optimizer)`,
  `scaler.update()` in place of the plain `loss.backward()`/`optimizer.step()`.
- **`evaluate_epoch`/`run_test_evaluation`** — wrap the forward pass in
  the same `autocast` context (no scaler needed — no backward pass
  happens during evaluation).
- When `use_amp=False` (CPU, or `mixed_precision: false`), `autocast(..., enabled=False)`
  is a no-op and `GradScaler(..., enabled=False)` degrades to plain
  `loss.backward()`/`optimizer.step()` — so this is one code path for
  both, not a fork.

Checkpoints record `"trained_with_amp": <bool>` for provenance only —
weights are always saved/loaded as float32 regardless, so a checkpoint
trained with AMP loads and runs identically to one trained without it.

Console output confirms which mode a run used:
```
Mixed precision (float16): enabled
```
or, on CPU with the flag left on:
```
Note: training.mixed_precision=true but device is CPU - running in float32 instead (float16 autocast needs CUDA).
Mixed precision (float16): disabled
```

---

## 12. Real experiment lineage: catastrophic forgetting, in practice

This section is the actual chain of experiments run on this project's
two real datasets, kept here because it's a worked example of §10's
"catastrophic forgetting" caveat actually happening, and of the fix
that worked (and one that didn't). Each config lives in `configs/`; the
full per-epoch history and test metrics for each run are archived in
`experiments/<run_name>/summary.json`.

**The two datasets**, both normalized via `scripts/normalize_dataset.py`,
zero filename overlap between them:
- **308K set**: `/home/<user>/ml_data/normalized` — 160,724 authentic +
  147,390 tampered.
- **New set**: `/home/<user>/ml_data/raw_new_normalized` (from this
  project's `data/raw`) — 19,787 authentic + 20,123 tampered.
- **Combined set**: `/home/<user>/ml_data/combined` — the two above
  *symlinked* into one `authentic/`+`tampered/` tree (348,024 files, no
  extra disk space), used by the combined-training runs below.

| Run (`run_name`) | What changed | Checkpoint (`models/saved/`) | Test on 308K | Test on New set |
|---|---|---|---|---|
| `tamper_resnet50_full_finetune` | Fully unfrozen (`freeze_stages: []`), trained on 308K, batch=32, manually stopped at epoch 3 | `tamper_resnet50_full_finetune_best.pth` | epoch-3 val_loss 0.6055 (early, not fully converged) | not tested |
| `tamper_resnet50_full_finetune_resumed_b96` | Continued from that epoch-3 checkpoint, batch 32→96 (LRs linearly scaled 3x), still fully unfrozen | **`tamper_resnet50_full_finetune_resumed_b96_best.pth`** (epoch 6) | Acc 85.77%, F1 84.42% | Recall **1.57%** (essentially blind) |
| `tamper_resnet50_frozen_backbone_new_data` → `..._continued` | Entire backbone frozen, only head (2,049 params) retrained on the **New set alone**, init from the row above | `tamper_resnet50_frozen_backbone_new_data_continued_best.pth` (epoch 22) | Authentic accuracy collapsed **92.69% → 2.67%** | Recall **99.74%** |
| `tamper_resnet50_frozen_backbone_combined` (head-only attempt) | Backbone frozen, head retrained on the **combined set**, init from `full_finetune_resumed_b96` | (overwritten by the next row - not kept) | Acc 81.93%, F1 79.91% (plateaued epoch 1, underfit) | not separately tested |
| `tamper_resnet50_frozen_backbone_combined` (layer4 unfrozen) | `layer4` + head trainable on the combined set | (overwritten by the next row - not kept) | test F1 77.77% (worse than the head-only attempt above) | not separately tested |
| `tamper_resnet50_layer3_layer4_combined` | `layer3` + `layer4` + head trainable, init from `full_finetune_resumed_b96` (**not** from either combined attempt above), on the combined set | `tamper_resnet50_layer3_layer4_combined_best.pth` (epoch 4) | Acc 89.77%, F1 89.05% (beats even the original 308K-only model) | Recall **11.66%** (still mostly blind) |

**Two distinct lessons, both confirmed empirically, not just in theory:**

1. **Freezing the backbone doesn't prevent catastrophic forgetting on its
   own.** Row 3 froze every conv weight and still lost the 308K dataset
   almost entirely (92.69% → 2.67% authentic accuracy) by retraining the
   head on the New set alone — the *features* stayed intact, but the
   *decision boundary* (which lives entirely in the head) drifted
   completely to whatever single distribution it was shown. "Frozen
   backbone" protects features, not decisions.

2. **Combining datasets doesn't automatically fix that if the datasets
   are unevenly sized.** Row 6's combined training still ends up
   dominated by the 308K set (88% of the combined tampered examples come
   from it), so despite training on the union, the model barely learned
   the New set's tampered patterns (11.66% recall) before early stopping
   found a compromise that favored the majority distribution. A truly
   balanced fix would need either upsampling the smaller dataset,
   downweighting the larger one in the loss, or a curriculum that forces
   more gradient signal from the minority set — none of which this
   pipeline does automatically today.

**Practical recommendation from this lineage**: for the "fully unfrozen
ResNet50 on the 308K dataset" checkpoint specifically (the base model
before any of the domain-adaptation experiments above), use:
```
models/saved/tamper_resnet50_full_finetune_resumed_b96_best.pth
```
(epoch 6, val_loss 0.3217, `freeze_stages: []`, `model_init_kwargs` in
the checkpoint confirms the architecture). It's also the checkpoint
every later `init_from_checkpoint` in this lineage reads from — the
common ancestor of every domain-adaptation attempt above.
