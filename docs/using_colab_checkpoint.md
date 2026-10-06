# Using a Checkpoint Across the Local Project and Colab

This file exists identically in **both** projects:
- `ml_project/docs/using_colab_checkpoint.md` (local)
- `ml_project_colab/docs/using_colab_checkpoint.md` (Colab)

because the workflow crosses between them in both directions: train
locally → use/continue on Colab, or train on Colab → use/continue
locally. Keep both copies in sync if you edit this.

---

## Why this works with zero conversion

A checkpoint saved by `scripts/train_tamper_resnet50.py` is a single
`.pth` file — a Python dict containing `model_state_dict` (the
weights) plus `model_init_kwargs` (`freeze_stages`, `freeze_bn`,
`dropout` — everything needed to reconstruct the exact same
architecture) and some metadata (epoch, val_loss, run_name,
inference_threshold, etc.). Nothing in that file is tied to which
machine or filesystem produced it.

This is portable between the local project and the Colab project
**specifically because `src/models/tamper_resnet50.py` is byte-for-byte
identical in both** (the Colab project's copy was made directly from
the local one — see that project's `docs/colab_training_guide.md`,
"What's reused"). As long as that stays true, any checkpoint from
either project loads correctly in the other with the exact same code:

```python
model = TamperResNet50(**checkpoint["model_init_kwargs"], pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
```

(this is literally what `scripts/predict_tamper_resnet50.py::load_model()`
and `scripts/evaluate_on_dataset.py::main()` already do — you don't
need to write this yourself, just point their `--model` flag at the
file).

---

## Colab → Local: bringing a Colab-trained checkpoint home

1. **Find the file on Drive.** The Colab template config saves to
   `/content/drive/MyDrive/tamper_resnet50_colab/models/saved/<run_name>_best.pth`
   (and `.../models/checkpoints/<run_name>_last.pth`). Open
   [drive.google.com](https://drive.google.com) and navigate there.
2. **Download it** (right-click → Download) to your local machine.
3. **Place it anywhere in the local project** — there's no required
   location, since every script takes `--model <path>` explicitly.
   The convention this project uses is `models/saved/<descriptive_name>.pth`,
   e.g.:
   ```bash
   mv ~/Downloads/tamper_resnet50_colab_run1_best.pth \
      /path/to/ml_project/models/saved/tamper_resnet50_colab_run1_best.pth
   ```
4. **Use it** with any of the local project's existing scripts, exactly
   like a locally-trained checkpoint:
   ```bash
   # single-image prediction
   python scripts/predict_tamper_resnet50.py \
     --image path/to/image.jpg \
     --model models/saved/tamper_resnet50_colab_run1_best.pth \
     --mode predict

   # evaluate against a whole dataset
   python scripts/evaluate_on_dataset.py \
     --model models/saved/tamper_resnet50_colab_run1_best.pth \
     --dataset-dir /home/<user>/ml_data/normalized \
     --report-dir results/reports/colab_run1_on_308k

   # continue training locally from where Colab left off
   # (set this in a new config's model.init_from_checkpoint, then:)
   python scripts/train_tamper_resnet50.py --config configs/my_continuation.yaml
   ```

No path editing inside the checkpoint file itself is needed — the
`.pth` file has no baked-in filesystem paths, only weights and the
small metadata dict described above.

---

## Local → Colab: taking a local checkpoint to Colab

1. **Upload the `.pth` file** to your Drive, e.g. into
   `My Drive/tamper_resnet50_colab/models/saved/`.
2. In `configs/tamper_resnet50_colab_template.yaml` (or a copy of it),
   set:
   ```yaml
   model:
     init_from_checkpoint: "/content/drive/MyDrive/tamper_resnet50_colab/models/saved/<the_file>.pth"
   ```
3. Give the run a new `experiment.run_name` and new `paths.*` (so you
   don't overwrite the checkpoint you're continuing from), then run
   the training notebook/script as usual.

---

## Sanity-checking a checkpoint before trusting it

Regardless of which direction it travelled, load it once and print
what it actually is before running anything expensive:

```python
import torch
ckpt = torch.load("path/to/checkpoint.pth", map_location="cpu")
print("run_name:", ckpt["run_name"])
print("epoch:", ckpt["epoch"], "val_loss:", ckpt["val_loss"])
print("architecture:", ckpt["model_init_kwargs"])
print("initialized_from:", ckpt.get("initialized_from"))
```

`model_init_kwargs` in particular tells you exactly what was frozen
(`freeze_stages`) when this checkpoint was produced — useful when
you've been running several variants and aren't 100% sure which file
is which (the `run_name` and `experiments/<run_name>/summary.json` are
the authoritative record either way).

## Caveat: this checkpoint format assumes matching code

If you ever change `src/models/tamper_resnet50.py`'s architecture in
one project without mirroring the change in the other (e.g. add a new
layer, change `embedding_dim`), checkpoints stop being portable
between them — `load_state_dict()` will raise a key-mismatch error.
Keep the two copies of that file identical (and of the other reused
files listed in `docs/colab_training_guide.md`) if you want this
cross-project portability to keep working.
