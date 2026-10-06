"""
ResNetCNN: image branch backed by a pretrained ResNet50 (ImageNet weights),
used in place of the from-scratch BaselineCNN/DeepCNN when a stable, strong
starting point matters more than "trained entirely from scratch".

Why this instead of a bigger from-scratch DeepCNN:
    DeepCNN (7 plain conv layers, no BatchNorm/skip connections, random
    init) turned out to be unstable to train on this dataset - it sat at
    0% recall for several epochs, briefly swung to 100% recall, then
    collapsed back to predicting one class for everything, and early
    stopping locked onto a checkpoint from before it ever "woke up".
    That's a known failure mode of deep plain CNNs trained from random
    initialization on a small dataset (~9500 images is small for a
    3M+ parameter net learning from scratch).

    A pretrained backbone sidesteps this: ResNet50's conv weights already
    encode useful, general image features from 1.2M ImageNet images, so
    training here only has to learn a new 128-dim embedding + 1 logit on
    top of already-sensible features, instead of learning everything
    (including "what an edge is") from random weights. Freezing the
    backbone (frozen by default - see `freeze_backbone`) means the vast
    majority of the network's weights can't move at all, which removes
    the instability we saw in DeepCNN by construction: there's much less
    that can go wrong when only ~260K parameters (the head) are being
    optimized instead of millions.

API-compatible with BaselineCNN/DeepCNN (extract_features(),
extract_conv_maps(), get_last_conv_layer(), embedding_dim, forward()
returning a raw logit) so scripts/train.py can switch between all three
by changing one config line, and this remains a valid image branch for
the later multi-modal model.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetCNN(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
        num_frozen_layers: int = 0,
    ):
        """
        freeze_backbone: True freezes the entire backbone (stem + layer1-4) -
            takes priority over num_frozen_layers when True.
        num_frozen_layers: when freeze_backbone=False, freezes the stem
            (conv1+bn1) plus the first N of ResNet50's 4 named stages
            (layer1..layer4), 0-4. E.g. num_frozen_layers=2 freezes the
            stem + layer1 + layer2, leaving layer3, layer4, and the head
            trainable - a middle ground between "fully frozen" (only the
            head learns) and "fully fine-tuned" (everything learns):
            early layers keep their generic ImageNet features fixed while
            later, more task-specific layers adapt to this dataset.
        """
        super().__init__()

        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)

        # Keep every conv stage (through layer4), drop torchvision's own
        # avgpool + fc - those are replaced with our own head below so the
        # single-logit binary-classification output matches BaselineCNN/DeepCNN.
        # Indices in this Sequential: 0=conv1, 1=bn1, 2=relu, 3=maxpool
        # (together, "the stem"), 4=layer1, 5=layer2, 6=layer3, 7=layer4.
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 2048, H, W) -> (B, 2048, 1, 1)
        # AdaptiveAvgPool2d means the flatten size is always 2048 regardless
        # of img_size - unlike BaselineCNN/DeepCNN, no dummy forward pass is
        # needed to compute it.
        backbone_out_features = backbone.fc.in_features  # 2048 for resnet50

        self.embedding_dim = 128
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_features, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.output_head = nn.Linear(self.embedding_dim, 1)

        self.freeze_backbone = freeze_backbone
        self.num_frozen_layers = 4 if freeze_backbone else max(0, min(4, num_frozen_layers))

        if freeze_backbone:
            self._frozen_modules = [self.features]
        elif num_frozen_layers > 0:
            stem = self.features[:4]
            frozen_stages = list(self.features[4:8])[:self.num_frozen_layers]
            self._frozen_modules = [stem] + frozen_stages
        else:
            self._frozen_modules = []

        for module in self._frozen_modules:
            for p in module.parameters():
                p.requires_grad = False
            module.eval()  # keep pretrained BatchNorm running stats fixed too

    def train(self, mode: bool = True):
        """Override so BatchNorm/Dropout inside any frozen portion of the
        backbone stay in eval mode even when the rest of the model is in
        train mode - otherwise BatchNorm would keep updating its running
        statistics from this small dataset despite requires_grad=False,
        which defeats the point of freezing it."""
        super().train(mode)
        for module in self._frozen_modules:
            module.eval()
        return self

    def freeze_description(self) -> str:
        if self.freeze_backbone:
            return "backbone fully frozen"
        if self.num_frozen_layers > 0:
            return f"stem + first {self.num_frozen_layers} of 4 stages frozen"
        return "full fine-tuning"

    def extract_conv_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial conv feature maps, shape (batch, 2048, 7, 7) at img_size=224."""
        return self.features(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """128-dim feature embedding, shape (batch, embedding_dim) - same
        hand-off point as BaselineCNN/DeepCNN.extract_features() for a
        future multi-modal fusion model."""
        conv_maps = self.extract_conv_maps(x)
        pooled = torch.flatten(self.global_pool(conv_maps), 1)
        return self.embedding_head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.output_head(features)
        return logits  # shape: (batch_size, 1) — raw logits

    def get_last_conv_layer(self) -> nn.Conv2d:
        """Reserved for Grad-CAM later - the last Conv2d in ResNet50's
        final stage (layer4's last bottleneck block)."""
        conv_layers = [m for m in self.features.modules() if isinstance(m, nn.Conv2d)]
        return conv_layers[-1]


def print_shape_trace(model: ResNetCNN, img_size: int = 224, in_channels: int = 3) -> None:
    device = next(model.parameters()).device
    x = torch.zeros(1, in_channels, img_size, img_size, device=device)

    print("Shape trace through ResNetCNN (batch dimension omitted):")
    print(f"  Input                                : {tuple(x.shape[1:])}")

    with torch.no_grad():
        conv_maps = model.extract_conv_maps(x)
        print(f"  extract_conv_maps() output (ResNet50 layer4): {tuple(conv_maps.shape[1:])}")

        pooled = torch.flatten(model.global_pool(conv_maps), 1)
        print(f"  after global average pool            : {tuple(pooled.shape[1:])}")

        embedding = model.embedding_head(pooled)
        print(f"  extract_features() output (embedding): {tuple(embedding.shape[1:])}")

        out = model.output_head(embedding)
        print(f"  Output (single logit)                : {tuple(out.shape[1:])}")
    print()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} total "
          f"({model.freeze_description()})")
    print()
