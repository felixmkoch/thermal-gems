from torch import nn
from functools import partial

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.models.mixer_seq_simple import create_block 
from mamba_ssm.models.mixer_seq_simple import _init_weights


class MambaBackbone(nn.Module):
    '''
    Backbone blocks of the MAMBA Model.
    '''
    def __init__(self,
                d_model: int,
                num_layers: int,
                ssm_config=None,
                norm_epsilon: float = 1e-5,
                rms_norm: bool = False,
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                dtype=None,
                attn_layer_idx=[],
                attn_config={},
                d_intermediate=0,
                use_mamba1=False
                ):
        
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        ssm_config = {"layer": "Mamba2"}
        if use_mamba1:
            ssm_config = {"layer": "Mamba1"}

        self.blocks = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_config,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_config,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def forward(self, x):
            
        hidden_states = x

        residual = None

        for block in self.blocks: 
            hidden_states, residual = block(hidden_states, residual)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        return hidden_states


class Mamba(nn.Module):

    def __init__(self,
                 num_features: int = 8,
                 d_model: int = 32,
                 nhid:int = 32,
                 num_layers: int = 1,
                 forecast_horizon: int = 1,
                 num_targets: int = 1,
                 use_mamba1: bool = False,
                 dtype=None,
                 **kwargs
                 ):

        super().__init__()

        self.num_features = num_features
        self.d_model = d_model
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.num_targets = num_targets
        self.use_mamba1 = use_mamba1 
        self.dtype = dtype


        self.mamba_backbone = MambaBackbone(
            d_model=d_model,
            num_layers=self.num_layers,
            norm_epsilon=1e-5,
            rms_norm=False,     # Doesn't work with true yet.
            use_mamba1=use_mamba1,
            dtype=self.dtype
        )

        self.encoder = nn.Linear(num_features, out_features=d_model)

        self.decoder = nn.Sequential(nn.Linear(d_model, nhid), nn.GELU(), nn.Linear(nhid, forecast_horizon * num_targets))

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **({}),
            )
        )



    def forward(self, x):

        x = self.encoder(x)

        # (B, S, F) -> (S, B, F) because Mamba handles it in this format.
        x = x.permute(1, 0, 2)

        hidden_states = self.mamba_backbone(x)

        # (S, B, F) -> (B, S, F)
        hidden_states = hidden_states.permute(1, 0, 2)

        hidden_states = hidden_states[:, -1, :]

        output = self.decoder(hidden_states)

        out = output.view(-1, self.forecast_horizon, self.num_targets)

        return out




