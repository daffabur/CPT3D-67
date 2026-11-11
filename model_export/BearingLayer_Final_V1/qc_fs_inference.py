"""Frozen QC-FS V3-D inference.

Input: preprocessed tensor [B, 9, Z, Y, X].
Output: qc [MPa], fs [kPa].

Does not import training scripts or modify checkpoints.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                channels, channels, 3,
                padding=dilation, dilation=dilation
            ),
            nn.LeakyReLU(0.1),
            nn.Conv3d(
                channels, channels, 3, padding=1
            ),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.net(x), 0.1)


class QCFSModel(nn.Module):
    def __init__(self, channels=16):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv3d(9, channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            Block(channels, 1),
            Block(channels, 2),
            Block(channels, 4),
        )

        self.qc_head = nn.Conv3d(
            channels, 1, 3, padding=1
        )
        self.fs_head = nn.Conv3d(
            channels, 1, 3, padding=1
        )

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 9:
            raise ValueError(
                "Expected input [B, 9, Z, Y, X]"
            )

        hidden = self.body(x)
        mask = x[:, 1:2]

        qc = x[:, 2:3] + self.qc_head(hidden)
        fs = x[:, 8:9] + self.fs_head(hidden)

        qc = torch.where(
            mask > 0, x[:, 0:1], qc
        )
        fs = torch.where(
            mask > 0, x[:, 7:8], fs
        )

        return qc, fs


class FrozenQCFS:
    def __init__(self, checkpoint=None, device="cpu"):
        if checkpoint is None:
            checkpoint = (
                Path(__file__).resolve().parent
                / "QC_FS_V3_D.pt"
            )

        self.device = torch.device(device)

        ckpt = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if ckpt["input_channels"] != 9:
            raise ValueError(
                "Unsupported checkpoint input channels"
            )

        self.qc_norm = ckpt["qc_normalization"]
        self.fs_norm = ckpt["fs_normalization"]

        self.model = QCFSModel(
            channels=int(ckpt["channels"])
        )

        self.model.load_state_dict(
            ckpt["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, features):
        """Returns qc MPa and fs kPa, [B,1,Z,Y,X]."""
        x = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )

        qc_normed, fs_normed = self.model(x)

        qc_log = (
            qc_normed * self.qc_norm["std"]
            + self.qc_norm["mean"]
        ).clamp(0.0, 6.0)

        qc_mpa = torch.expm1(qc_log)

        fs_log = (
            fs_normed * self.fs_norm["std"]
            + self.fs_norm["mean"]
        )

        fs_kpa = torch.expm1(fs_log).clamp_min(0)

        return (
            qc_mpa.cpu().numpy(),
            fs_kpa.cpu().numpy(),
        )
