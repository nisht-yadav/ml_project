"""
TamperResNet50: config-driven ResNet50 binary tampering classifier.

This is a separate model class from src/models/resnet_model.py's
ResNetCNN (which stays as-is for the existing baseline/deep/resnet50
comparison pipeline in scripts/train.py). ResNetCNN pools down to a
128-dim embedding before its output head, which hides the standard
2048-dim penultimate ResNet50 feature vector this task needs to expose
for later multi-agent feature fusion. Everything else here (frozen-BN
handling, differential LR groups, extract_features) is also exposed
directly as constructor/config options instead of being baked in, per
the brief.

Design summary:
- `freeze_stages`: explicit list of named stages to freeze
  ("conv1", "layer1", "layer2", "layer3", "layer4") - never an ambiguous
  frozen-layer count.
- `freeze_bn`: freezes BatchNorm running statistics (eval mode) across
  the WHOLE backbone independently of `freeze_stages`. Weight-freezing
  (requires_grad) and BN running-stat-freezing (train()/eval() mode)
  are different mechanisms - see the freeze_bn docstring below.
- `get_param_groups(lr_groups)`: builds optimizer param groups from a
  {group_name: lr} dict, so differential LRs are a config value, not
  hardcoded optimizer construction in the training script.
- `forward(x, return_features=False)`: raw logits, optionally paired
  with the 2048-d pre-FC feature vector - no separate architecture
  needed later to get both.
- `predict` / `extract_features` / `predict_with_features`: the three
  call modes a saved checkpoint needs to support to be reusable
  directly as Agent A/B in a later fusion setup (see class docstring
  point 8 in the brief) without rewriting inference code.

Loss-side vs. inference-side bias (documented here since both levers
live partly in this file, partly in the training script):
    - `pos_weight` (passed to nn.BCEWithLogitsLoss by the training
      script, NOT stored on this model) biases TRAINING: it scales the
      loss contribution of positive (tampered) examples, so gradients
      push harder on mistakes involving that class. This changes what
      the model learns.
    - `threshold` (used only by `predict`/`predict_with_features` below,
      and by the training script's final test-set report) biases
      INFERENCE: it moves the decision boundary applied to an already-
      trained model's sigmoid output. This changes nothing about the
      model's weights - it can be changed freely, per-call, with no
      retraining.
    These are independent levers. Use one, the other, both, or neither.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

STAGE_NAMES: Tuple[str, ...] = ("conv1", "layer1", "layer2", "layer3", "layer4")
GROUP_NAMES: Tuple[str, ...] = STAGE_NAMES + ("head",)


class TamperResNet50(nn.Module):
    def __init__(
        self,
        freeze_stages: Optional[Sequence[str]] = None,
        freeze_bn: bool = False,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        """
        freeze_stages: list of stage names to freeze (requires_grad=False),
            drawn from STAGE_NAMES = ("conv1", "layer1", "layer2", "layer3",
            "layer4"). "conv1" means the full stem (conv1+bn1+relu+maxpool),
            matching torchvision's own layer naming for the other 4.
            e.g. ["conv1", "layer1"] freezes only the stem + layer1,
            leaving layer2/3/4 + head trainable. [] or None = nothing
            frozen (full fine-tuning).
        freeze_bn: if True, EVERY BatchNorm2d in the backbone (including
            ones in stages that are otherwise trainable) is kept in eval
            mode, so its running_mean/running_var stop updating from
            this dataset's batch statistics. This is separate from
            `freeze_stages`: a stage can have trainable conv weights
            while its BN running stats stay frozen, which matters when
            batch_size is small enough that fresh BN statistics would be
            noisy. Stages listed in `freeze_stages` always have their BN
            put in eval mode regardless of this flag (freezing the
            weights but letting BN stats keep drifting would be
            inconsistent).
        dropout: dropout applied to the pooled 2048-d feature vector
            before the final linear layer.
        pretrained: True loads ImageNet weights (the normal case). False
            is only useful for architecture testing.
        """
        super().__init__()

        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = backbone.fc.in_features  # 2048 for resnet50
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.feature_dim, 1)

        self._stage_modules: Dict[str, nn.Module] = {
            "conv1": self.stem,
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
        }

        self.freeze_stages_cfg: List[str] = list(freeze_stages or [])
        self.freeze_bn_cfg: bool = freeze_bn

        unknown = set(self.freeze_stages_cfg) - set(STAGE_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown stage name(s) in freeze_stages: {sorted(unknown)}. "
                f"Valid stage names are: {list(STAGE_NAMES)}"
            )

        self._apply_freeze_stages()

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def _apply_freeze_stages(self) -> None:
        for name, module in self._stage_modules.items():
            requires_grad = name not in self.freeze_stages_cfg
            for p in module.parameters():
                p.requires_grad = requires_grad

    def train(self, mode: bool = True):
        """
        Overridden so BatchNorm running statistics stay frozen exactly
        where the config says they should, independent of whichever
        stages have trainable weights:
          - any stage in freeze_stages is always put back in eval mode
            (frozen weights + drifting BN stats would be inconsistent).
          - if freeze_bn=True, EVERY backbone BatchNorm2d is put in eval
            mode, even inside stages whose conv weights are training.
        """
        super().train(mode)

        for name, module in self._stage_modules.items():
            if name in self.freeze_stages_cfg:
                module.eval()

        if self.freeze_bn_cfg:
            for module in self._stage_modules.values():
                for m in module.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.eval()

        return self

    def freeze_description(self) -> str:
        frozen = ", ".join(self.freeze_stages_cfg) if self.freeze_stages_cfg else "none"
        return f"frozen stages: [{frozen}] | freeze_bn: {self.freeze_bn_cfg}"

    # ------------------------------------------------------------------
    # Differential learning rates
    # ------------------------------------------------------------------
    def get_param_groups(self, lr_groups: Dict[str, float]) -> List[dict]:
        """
        Builds torch optimizer param groups from a config dict such as:
            {"conv1_layer1": 1e-5, "layer2": 3e-5, "layer3": 1e-4,
             "layer4": 3e-4, "head": 1e-3}

        Each key is one or more stage names (from GROUP_NAMES) joined by
        "_", so multiple stages can share one LR ("conv1_layer1") or
        each stage can get its own key. Every stage that still has
        trainable parameters after `freeze_stages` was applied MUST be
        covered by exactly one key - this is enforced (raises
        ValueError) instead of silently defaulting some stage to no LR
        / the wrong LR, since a silent gap here would train part of the
        network with an LR nobody chose.
        """
        stage_to_key: Dict[str, str] = {}
        for key in lr_groups:
            for stage in key.split("_"):
                if stage not in GROUP_NAMES:
                    raise ValueError(
                        f"Unknown stage name '{stage}' in lr_groups key '{key}'. "
                        f"Valid stage names are: {list(GROUP_NAMES)}"
                    )
                if stage in stage_to_key:
                    raise ValueError(
                        f"Stage '{stage}' appears in more than one lr_groups key "
                        f"('{stage_to_key[stage]}' and '{key}')."
                    )
                stage_to_key[stage] = key

        trainable_stages = [s for s in STAGE_NAMES if s not in self.freeze_stages_cfg] + ["head"]
        missing = [s for s in trainable_stages if s not in stage_to_key]
        if missing:
            raise ValueError(
                f"lr_groups does not cover trainable stage(s) {missing}. "
                f"Every stage not listed in freeze_stages needs an entry "
                f"(alone or combined with '_') in lr_groups."
            )

        param_groups = []
        for key, lr in lr_groups.items():
            params = []
            for stage, owner_key in stage_to_key.items():
                if owner_key != key:
                    continue
                module = self.fc if stage == "head" else self._stage_modules[stage]
                params.extend(p for p in module.parameters() if p.requires_grad)
            if params:
                param_groups.append({"params": params, "lr": lr, "name": key})

        return param_groups

    # ------------------------------------------------------------------
    # Forward / feature extraction
    # ------------------------------------------------------------------
    def _pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)  # (batch, 2048)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Raw logits, shape (batch, 1) - no sigmoid applied (matches
        BCEWithLogitsLoss, which expects raw logits for numerical
        stability). Pass return_features=True to also get back the
        2048-d pooled feature vector (post-GAP, pre-FC) in one forward
        pass, for callers that need both without running the backbone
        twice.
        """
        features = self._pooled_features(x)
        logits = self.fc(self.dropout(features))
        if return_features:
            return logits, features
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        2048-d penultimate feature vector (post global-average-pool,
        pre-FC), independent of the classification head - this is the
        hand-off point for later multi-agent fusion (Agent A/B): this
        same trained model can supply features here without touching
        (or even loading) the FC layer's semantics.
        """
        self.eval()
        with torch.no_grad():
            return self._pooled_features(x)

    # ------------------------------------------------------------------
    # Inference call modes (see module docstring: pos_weight vs. threshold)
    # ------------------------------------------------------------------
    def predict(self, x: torch.Tensor, threshold: float = 0.5):
        """
        Inference-only call: applies sigmoid to the raw logits and a
        decision threshold (default 0.5, override freely without
        retraining - see module docstring).

        Returns (labels, confidences, logits):
            labels:      LongTensor (batch,), 1 = tampered, 0 = authentic
            confidences: FloatTensor (batch,), sigmoid probability of
                         "tampered" (NOT "probability of the predicted
                         class" - always P(tampered), so it's directly
                         comparable across calls regardless of the
                         predicted label)
            logits:      FloatTensor (batch,), raw pre-sigmoid scores,
                         kept for downstream models/debugging
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x).squeeze(-1)
            probs = torch.sigmoid(logits)
            labels = (probs >= threshold).long()
        return labels, probs, logits

    def predict_with_features(self, x: torch.Tensor, threshold: float = 0.5):
        """
        Same as predict(), plus the 2048-d feature vector in the same
        forward pass - the single call a fusion setup needs to use this
        model as Agent A/B: prediction, confidence, and features
        together, nothing re-architected.

        Returns (labels, confidences, logits, features).
        """
        self.eval()
        with torch.no_grad():
            logits, features = self.forward(x, return_features=True)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            labels = (probs >= threshold).long()
        return labels, probs, logits, features

    def get_last_conv_layer(self) -> nn.Conv2d:
        """Reserved for Grad-CAM: the last Conv2d in layer4's final block."""
        conv_layers = [m for m in self.layer4.modules() if isinstance(m, nn.Conv2d)]
        return conv_layers[-1]


def print_shape_trace(model: TamperResNet50, img_size: int = 224, in_channels: int = 3) -> None:
    device = next(model.parameters()).device
    x = torch.zeros(1, in_channels, img_size, img_size, device=device)

    print("Shape trace through TamperResNet50 (batch dimension omitted):")
    print(f"  Input                       : {tuple(x.shape[1:])}")

    with torch.no_grad():
        features = model._pooled_features(x)
        print(f"  extract_features() output   : {tuple(features.shape[1:])}")
        logits = model.fc(model.dropout(features))
        print(f"  Output (single logit)       : {tuple(logits.shape[1:])}")
    print()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} total ({model.freeze_description()})")
    print()
