"""
BaselineCNN: a simple CNN built from scratch (no pretrained weights) for
binary image-tampering classification.

Architecture:
    Input  3 x H x W
    [Conv2d -> ReLU -> MaxPool2d]  x 3          (self.features)
    Flatten -> Linear -> ReLU -> Dropout        (self.embedding_head)
    Linear(1)                                    (self.output_head)

Design notes for later stages of the project:
- The head is split into `self.embedding_head` (produces a 128-dim
  feature vector) and `self.output_head` (turns that vector into the
  single classification logit), instead of one fused block. This is
  what makes the image features usable outside this classifier: for a
  future multi-modal model (image CNN + DCT/ELA features), you can call
  `extract_features(x)` to get that 128-dim vector for a given batch of
  images and concatenate it with frequency-domain features before
  feeding a separate fusion classifier — without touching this class or
  retraining it as a black box. Nothing is extracted or stored by
  default; these are just methods available when needed.
- `self.features` is kept separate for the same reason at the
  convolutional level: `extract_conv_maps()` returns the raw spatial
  feature maps (before flattening), useful if a later approach wants
  spatial features rather than a flat vector. The last Conv2d layer
  inside it is also what Grad-CAM will need to hook into - see
  `get_last_conv_layer()`.
- The flatten size is computed automatically from a dummy forward pass
  instead of being hard-coded, so changing IMG_SIZE never silently
  breaks the Linear layer's input size.
- A single output neuron (`nn.Linear(..., 1)`) is used because this is
  binary classification: one logit is enough to represent
  P(tampered) via a sigmoid, rather than two logits + softmax.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, img_size: int = 224, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 3xHxW -> 32xHxW (padding=1 keeps spatial size)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                           # -> 32 x H/2 x W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # -> 64 x H/2 x W/2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                           # -> 64 x H/4 x W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # -> 128 x H/4 x W/4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                           # -> 128 x H/8 x W/8
        )

        flat_features = self._infer_flatten_size(in_channels, img_size)
        self.embedding_dim = 128  # size of the feature vector extract_features() returns

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Single output neuron: raw logit for "tampered". BCEWithLogitsLoss
        # is used with this, so NO sigmoid is applied here - see the note
        # in scripts/train.py.
        self.output_head = nn.Linear(self.embedding_dim, 1)

    def _infer_flatten_size(self, in_channels: int, img_size: int) -> int:
        """Run one dummy image through `self.features` to find out how
        many values come out the other end, instead of hand-calculating
        (and risking a mismatch if img_size or the architecture changes)."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            out = self.features(dummy)
        return out.numel()

    def extract_conv_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial conv feature maps, shape (batch, 128, H/8, W/8), taken
        before flattening. Not used by the baseline classifier flow
        itself - available for approaches that want spatial (not just
        vector) image features, e.g. Grad-CAM later."""
        return self.features(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        The 128-dim feature embedding for a batch of images, taken right
        before the final classification layer, shape (batch, embedding_dim).

        This is the intended hand-off point for a future multi-modal
        model: run your DCT/ELA feature extraction separately, concatenate
        it with this vector, and feed the combined vector into a new
        fusion classifier - this CNN doesn't need to change to support that.
        """
        return self.embedding_head(self.extract_conv_maps(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.output_head(features)
        return logits  # shape: (batch_size, 1) — raw logits, not probabilities

    def get_last_conv_layer(self) -> nn.Conv2d:
        """
        Returns the final Conv2d layer of the feature extractor.
        Not used yet - reserved for Grad-CAM in a later stage of the
        project, which needs the activations/gradients of the last
        convolutional layer.
        """
        conv_layers = [m for m in self.features if isinstance(m, nn.Conv2d)]
        return conv_layers[-1]


def print_shape_trace(model: BaselineCNN, img_size: int = 224, in_channels: int = 3) -> None:
    """
    Pushes a dummy image through the model and prints the tensor shape
    after every conv block, the extractable feature stages, and the
    final output - using the model's ACTUAL computed shapes rather than
    assumed ones.
    """
    device = next(model.parameters()).device
    x = torch.zeros(1, in_channels, img_size, img_size, device=device)

    print("Shape trace through BaselineCNN (batch dimension omitted):")
    print(f"  Input                                : {tuple(x.shape[1:])}")

    block = 1
    with torch.no_grad():
        for layer in model.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                print(f"  after conv block {block}                 : {tuple(x.shape[1:])}")
                block += 1

        print(f"  extract_conv_maps() output           : {tuple(x.shape[1:])}")

        for layer in model.embedding_head:
            x = layer(x)
        print(f"  extract_features() output (embedding): {tuple(x.shape[1:])}")

        out = model.output_head(x)
        print(f"  Output (single logit)                : {tuple(out.shape[1:])}")
    print()
