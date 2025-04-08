from typing import Dict, Tuple

from loguru import logger
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange
from torch.optim import Adam

from c4a0_rust import N_COLS, N_ROWS  # type: ignore


class ModelConfig(BaseModel):
    """Configuration for OptimizedConnectFourNet."""

    n_residual_blocks: int = 6
    base_filters: int = 64
    policy_hidden_size: int = 128
    value_hidden_size: int = 128
    # conv_filter_size: int
    n_policy_layers: int
    n_value_layers: int
    lr_schedule: Dict[int, float]
    l2_reg: float = 1e-4


class ResidualBlock(nn.Module):
    """Residual Block với 2 lớp conv và skip connection tối ưu"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        return F.relu(x + self.conv(x), inplace=True)


class ConnectFourNet(pl.LightningModule):
    EPS = 1e-8

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lr_schedule = config.lr_schedule
        self.l2_reg = config.l2_reg

        # Input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, config.base_filters, 3, padding=1),
            nn.BatchNorm2d(config.base_filters),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(config.base_filters)
            for _ in range(config.n_residual_blocks)
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(config.base_filters, 2, 1),
            nn.Flatten(),
            nn.Linear(2 * N_ROWS * N_COLS, config.policy_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.policy_hidden_size, N_COLS),
            nn.LogSoftmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.base_filters, config.value_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_hidden_size, 2),
            nn.Tanh()
        )

        # Metrics
        self.policy_kl_div = torchmetrics.KLDivergence(log_prob=True)
        self.q_penalty_mse = torchmetrics.MeanSquaredError()
        self.q_no_penalty_mse = torchmetrics.MeanSquaredError()

        self.save_hyperparameters(config.model_dump())

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        x = self.res_blocks(x)
        
        policy_logprobs = self.policy_head(x)
        q_values = self.value_head(x)
        
        return policy_logprobs, q_values[:, 0], q_values[:, 1]

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        pos = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            policy, q_penalty, q_no_penalty = self.forward(pos)
        policy = policy.contiguous().cpu().numpy()
        q_penalty = q_penalty.contiguous().cpu().numpy()
        q_no_penalty = q_no_penalty.contiguous().cpu().numpy()
        return policy, q_penalty, q_no_penalty

    def configure_optimizers(self):
        gen_n: int = self.trainer.gen_n  # type: ignore
        assert gen_n is not None, "please pass gen_n to trainer"
        schedule = sorted(list(self.lr_schedule.items()))
        _, lr = schedule.pop(0)
        for gen_threshold, gen_rate in schedule:
            if gen_n < gen_threshold:
                break
            lr = gen_rate

        logger.info("using lr {} for gen_n {}", lr, gen_n)
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, log_prefix="val")

    def step(self, batch, log_prefix):
        pos, policy_target, q_penalty_target, q_no_penalty_target = batch
        policy_logprob, q_penalty_pred, q_no_penalty_pred = self.forward(pos)
        policy_logprob_targets = torch.log(policy_target + self.EPS)

        policy_loss = self.policy_kl_div(policy_logprob_targets, policy_logprob)
        q_penalty_loss = self.q_penalty_mse(q_penalty_pred, q_penalty_target)
        q_no_penalty_loss = self.q_no_penalty_mse(
            q_no_penalty_pred, q_no_penalty_target
        )
        loss = policy_loss + q_penalty_loss + q_no_penalty_loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        self.log(f"{log_prefix}_policy_kl_div", policy_loss)
        self.log(f"{log_prefix}_value_mse", q_penalty_loss)
        return loss