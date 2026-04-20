"""1D ResNet-18 for cyclic voltammogram mechanism classification.

Adapts the standard ResNet-18 architecture (He et al., 2016) for 1D
electrochemical signals following Hoar et al. (2022).  Input tensors have
shape ``(batch, in_channels, seq_len)`` where *in_channels* encodes the
forward/reverse current channels across multiple scan rates and *seq_len*
is the number of discretised potential points.

Registered as ``resnet1d_18`` in the model registry.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.registry import BaseModel, register


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class BasicBlock1D(nn.Module):
    """Standard ResNet basic block adapted for 1D convolutions.

    Two conv layers with batch-norm and a residual skip connection.
    A 1x1 projection shortcut is used when the spatial or channel
    dimensions change.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# ResNet1D-18
# ---------------------------------------------------------------------------


@register("resnet1d_18")
class ResNet1D18(BaseModel):
    """1D ResNet-18 for mechanism classification from cyclic voltammograms.

    Architecture mirrors the standard ResNet-18 but uses 1D convolutions.
    The initial conv layer uses a wider kernel (7) to capture broad
    electrochemical features, followed by four residual stages and a
    global average pool before the classification head.

    Args:
        in_channels: Number of input channels.  For the paper's format
            this is ``num_scan_rates * 3`` (3 channels per scan rate) or
            simply 3 if scan rates are stacked along the batch/sample
            dimension.  Default ``3``.
        num_classes: Number of mechanism classes.  Default ``5``
            (E, EC, CE, ECE, DISP1).
        base_channels: Width of the first residual stage.  Subsequent
            stages double this.  Default ``64``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        base_channels: int = 64,
        zero_init_residual: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_planes = base_channels

        # Initial convolution (wider kernel for electrochemical features)
        self.conv1 = nn.Conv1d(
            in_channels, base_channels, kernel_size=7,
            stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages: [2, 2, 2, 2] blocks = 16 conv layers + initial = ~18
        self.layer1 = self._make_layer(base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8 * BasicBlock1D.expansion, num_classes)

        self.init_weights()

        # Zero-gamma: set bn2.weight=0 in each BasicBlock so residual
        # branches output zero at init → blocks start as identity mappings.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.zeros_(m.bn2.weight)

    def _make_layer(
        self, planes: int, num_blocks: int, stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_planes, planes * BasicBlock1D.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )

        layers = [BasicBlock1D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * BasicBlock1D.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(self.in_planes, planes))

        return nn.Sequential(*layers)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim features before the classification head.

        Args:
            x: Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns:
            Feature tensor of shape ``(batch, base_channels * 8)``.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        features = self.get_features(x)
        return self.fc(features)
