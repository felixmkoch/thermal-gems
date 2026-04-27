import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class XLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,             # NEW: input feature size
        embedding_dim: int = 128,
        context_length: int = 256,
        num_blocks: int = 7,
        num_heads: int = 4,
        conv1d_kernel_size: int = 4,
        proj_factor: float = 1.3,
        act_fn: str = "gelu",
        slstm_at: list[int] = [1],
        backend: str = "cuda",
        bias_init: str = "powerlaw_blockdependent",
        forecast_horizon: int = 1,
        num_targets: int = 1,
        **kwargs
    ):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.num_targets = num_targets
        self.embedding_dim = embedding_dim

        # Project input features to embedding_dim
        self.input_proj = nn.Linear(num_features, embedding_dim)

        # Build xLSTM config
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size,
                    qkv_proj_blocksize=num_heads,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init=bias_init,
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=proj_factor,
                    act_fn=act_fn,
                ),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=slstm_at,
        )

        self.model = xLSTMBlockStack(cfg)

        # Projection layer for forecasting
        self.fc = nn.Linear(embedding_dim, forecast_horizon * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, num_features)
        x = self.input_proj(x)  # → (batch, seq_len, embedding_dim)

        out = self.model(x)     # → (batch, seq_len, embedding_dim)

        # take the last time step
        out = out[:, -1, :]     # (batch, embedding_dim)

        # project to forecasting output
        out = self.fc(out)      # (batch, forecast_horizon * num_targets)

        # reshape to (batch, horizon, targets)
        out = out.view(-1, self.forecast_horizon, self.num_targets)

        return out
