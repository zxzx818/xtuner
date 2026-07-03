import math
from collections.abc import Callable

import torch
import torch.nn as nn
from dataclasses import dataclass
from xtuner.v1.model.base import XTunerBaseModelConfig
from xtuner.v1.model import BaseModel
from typing import ClassVar

from xtuner.v1.utils import get_device, get_torch_device_module
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from typing_extensions import override
from tqdm import tqdm
from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from xtuner.v1.model.utils.checkpointing import checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
_TOLERANCE = 1e-6


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


class InternS2PreviewTimeSeriesForecasterConfig(XTunerBaseModelConfig):
    """Configuration for :class:`InternS2PreviewTimeSeriesForecaster`.

    The Forecaster backbone dims (model_dims=1280, patch_len=32, 20 layers, ...) are
    fixed by Forecaster 2.5 200M and are not configurable here.
    """

    model_type: ClassVar[str] = "interns2_preview_time_series_forecaster"

    # Forecaster 2.5 200M is fixed at model_dims = 1280.
    TIMESFM_MODEL_DIMS: ClassVar[int] = 1280

    d_llm: int = 2560
    d_ts_encoder: int = 1024
    qformer_hidden_dim: int = 1280
    qformer_num_query_tokens: int = 32
    qformer_num_heads: int = 8
    qformer_num_layers: int = 2
    qformer_dropout: float = 0.0
    use_horizon_head: bool = True
    horizon_max_length: int = 0
    use_cross_attn_gate: bool = True
    cross_attn_kv_dim: int = 1280
    max_context: int = 2048
    max_horizon: int = 1024
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True
    return_backcast: bool = False
    default_pred_len: int = 720
    point_loss_weight: float = 1.0
    quantile_loss_weight: float = 1.0
    horizon_loss_weight: float = 1.0
    future_covariate_injection: str = "none"
    future_covariate_target_channel_idx: int = 0
    future_covariate_patch_len: int = 32
    future_covariate_queries_per_patch: int = 4
    future_covariate_hidden_dim: int = 1280
    future_covariate_num_heads: int = 8
    future_covariate_num_layers: int = 1
    future_covariate_dropout: float = 0.0
    future_covariate_max_horizon: int = 1024

    def build(self):
        return InternS2PreviewTimeSeriesForecaster(self)


class RMSNorm(nn.Module):
    """RMS normalization."""

    def __init__(self, num_features: int, *, epsilon: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and a linear residual connection."""

    def __init__(self, input_dims, hidden_dims, output_dims, use_bias, activation="swish"):
        super().__init__()
        self.hidden_layer = nn.Linear(
            in_features=input_dims,
            out_features=hidden_dims,
            bias=use_bias,
        )
        self.output_layer = nn.Linear(
            in_features=hidden_dims,
            out_features=output_dims,
            bias=use_bias,
        )
        self.residual_layer = nn.Linear(
            in_features=input_dims,
            out_features=output_dims,
            bias=use_bias,
        )
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation: {activation} not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(
            self.activation(self.hidden_layer(x))
        ) + self.residual_layer(x)


@dataclass(frozen=False)
class DecodeCache:
    """Cache for decoding."""

    next_index: torch.Tensor
    num_masked: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    kv_mask: torch.Tensor | None = None  # (B, cache_size), True=padded/masked


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(
        self,
        inputs: torch.Tensor,
        position: torch.Tensor | None = None,
    ):
        """Generates a JTensor of sinusoids with different frequencies."""
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding"
                "must match the hidden dimension of the inputs."
            )
        half_embedding_dim = self.embedding_dims // 2
        fraction = (
            2
            * torch.arange(0, half_embedding_dim, device=inputs.device)
            / self.embedding_dims
        )
        timescale = (
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        ).to(inputs.device)
        if position is None:
            seq_length = inputs.shape[1]
            position = torch.arange(seq_length, dtype=torch.float32, device=inputs.device)[
                None, :
            ]

        if len(inputs.shape) == 4:
            position = position[..., None, None]
            timescale = timescale[None, None, None, :]
        elif len(inputs.shape) == 3:
            position = position[..., None]
            timescale = timescale[None, None, :]
        else:
            raise ValueError("Inputs must be of rank 3 or 4.")

        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat([first_part, second_part], dim=-1)


class PerDimScale(nn.Module):
    """Per-dimension scaling."""

    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_factor = 1.442695041 / math.sqrt(self.num_dims) * nn.functional.softplus(self.per_dim_scale)
        return x * scale_factor


def _torch_dot_product_attention(query, key, value, mask=None):
    """Same (unscaled) attention as _dot_product_attention, fused kernel."""
    safe_mask = mask
    fully_masked_rows = None
    if mask is not None:
        # F.scaled_dot_product_attention can emit NaNs/NaN-grads when a query row is
        # fully masked. Keep the fused kernel by opening a single dummy key for
        # those rows, then zero their outputs.
        fully_masked_rows = ~mask.any(dim=-1, keepdim=True)
        if fully_masked_rows.any():
            dummy_key_mask = torch.zeros_like(mask)
            dummy_key_mask[..., 0] = True
            safe_mask = mask | (fully_masked_rows & dummy_key_mask)

    attention_dtype = value.dtype
    if query.dtype != attention_dtype:
        query = query.to(attention_dtype)
    if key.dtype != attention_dtype:
        key = key.to(attention_dtype)

    # 1. Permute inputs from (B, L, H, D) to the expected (B, H, L, D)
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    # 2. Fused attention kernel with scale=1.0 (disable 1/sqrt(d_k) scaling).
    output = nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=safe_mask, scale=1.0
    )

    if fully_masked_rows is not None and fully_masked_rows.any():
        output = output.masked_fill(fully_masked_rows, 0.0)

    # 3. Permute back to (B, L, H, D)
    output = output.permute(0, 2, 1, 3)
    return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        *,
        use_per_dim_scale: bool = True,
        use_rotary_position_embeddings: bool = True,
        use_bias: bool = False,
        attention_fn: Callable[..., torch.Tensor] = _torch_dot_product_attention,
        qk_norm: str = "rms",
        fuse_qkv: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.qk_norm = qk_norm
        self.fuse_qkv = fuse_qkv

        if self.in_features % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.in_features}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        if self.fuse_qkv:
            self.qkv_proj = nn.Linear(self.in_features, 3 * self.in_features, bias=use_bias)
        else:
            self.query = nn.Linear(self.in_features, self.in_features, bias=use_bias)
            self.key = nn.Linear(self.in_features, self.in_features, bias=use_bias)
            self.value = nn.Linear(self.in_features, self.in_features, bias=use_bias)
        self.out = nn.Linear(self.in_features, self.in_features, bias=use_bias)

        if self.qk_norm == "rms":
            self.query_ln = RMSNorm(self.head_dim)
            self.key_ln = RMSNorm(self.head_dim)
        else:
            self.query_ln = nn.Identity()
            self.key_ln = nn.Identity()

        self.use_rotary_position_embeddings = use_rotary_position_embeddings
        if self.use_rotary_position_embeddings:
            self.rotary_position_embedding = RotaryPositionalEmbedding(
                embedding_dims=self.head_dim,
            )

        self.use_per_dim_scale = use_per_dim_scale
        if use_per_dim_scale:
            self.per_dim_scale = PerDimScale(num_dims=self.head_dim)

    def make_attn_mask(
        self,
        query_length: int,
        num_all_masked_kv: torch.Tensor,
        query_index_offset: torch.Tensor | None = None,
        kv_length: int = 0,
    ) -> torch.Tensor:
        """Makes attention mask."""
        if kv_length == 0:
            kv_length = query_length

        q_index = torch.arange(query_length, device=num_all_masked_kv.device)[None, None, :, None]
        if query_index_offset is not None:
            q_index = q_index + query_index_offset[:, None, None, None]
        kv_index = torch.arange(kv_length, device=num_all_masked_kv.device)[None, None, None, :]
        return torch.logical_and(
            q_index >= kv_index,
            kv_index >= num_all_masked_kv[:, None, None, None],
        )

    def forward(
        self,
        inputs_q: torch.Tensor,
        *,
        decode_cache: DecodeCache | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecodeCache | None]:
        b, n_patches, _ = inputs_q.shape
        if patch_mask is None:
            patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)

        if self.fuse_qkv:
            qkv = self.qkv_proj(inputs_q)
            query, key, value = torch.chunk(qkv, 3, dim=-1)
            query = query.view(b, n_patches, self.num_heads, self.head_dim)
            key = key.view(b, n_patches, self.num_heads, self.head_dim)
            value = value.view(b, n_patches, self.num_heads, self.head_dim)
        else:
            query = self.query(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
            key = self.key(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
            value = self.value(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)

        if decode_cache is None:
            # Count only LEADING masked patches (not total). make_attn_mask assumes
            # all masked positions are a contiguous left-padding block.
            is_valid = ~patch_mask
            has_valid = is_valid.any(dim=-1)
            first_valid = torch.argmax(is_valid.to(torch.int32), dim=-1)
            seq_len = torch.tensor(patch_mask.shape[-1], dtype=torch.int32, device=patch_mask.device)
            num_masked = torch.where(has_valid, first_valid, seq_len)
            next_index = torch.zeros_like(num_masked, dtype=torch.int32)
        else:
            is_valid = ~patch_mask
            has_valid = is_valid.any(dim=-1)
            first_valid = torch.argmax(is_valid.to(torch.int32), dim=-1)
            seq_len = torch.tensor(patch_mask.shape[-1], dtype=torch.int32, device=patch_mask.device)
            leading_masked = torch.where(has_valid, first_valid, seq_len)
            num_masked = leading_masked + decode_cache.num_masked
            next_index = decode_cache.next_index.clone()

        if self.use_rotary_position_embeddings:
            position = (
                torch.arange(n_patches, device=inputs_q.device)[None, :]
                + next_index[:, None]
                - num_masked[:, None]
            )
            query = self.rotary_position_embedding(query, position)
            key = self.rotary_position_embedding(key, position)

        query = self.query_ln(query)
        key = self.key_ln(key)

        if self.use_per_dim_scale:
            query = self.per_dim_scale(query)

        if decode_cache is not None:
            _, decode_cache_size, _, _ = decode_cache.value.shape

            start = decode_cache.next_index[0]
            end = start + n_patches

            decode_cache.key[:, start:end] = key
            decode_cache.value[:, start:end] = value

            if decode_cache.kv_mask is None:
                decode_cache.kv_mask = torch.ones(
                    b, decode_cache_size, dtype=torch.bool, device=patch_mask.device,
                )
            decode_cache.kv_mask[:, start:end] = patch_mask

            key = decode_cache.key
            value = decode_cache.value
            decode_cache.next_index += n_patches
            decode_cache.num_masked = num_masked
            attn_mask = self.make_attn_mask(
                query_length=n_patches,
                num_all_masked_kv=num_masked,
                query_index_offset=next_index,
                kv_length=decode_cache_size,
            )
            kv_valid = ~decode_cache.kv_mask  # (B, decode_cache_size)
            attn_mask = attn_mask & kv_valid[:, None, None, :]
        else:
            attn_mask = self.make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)
            kv_valid = ~patch_mask  # (B, n_patches), True=valid
            attn_mask = attn_mask & kv_valid[:, None, None, :]

        x = self.attention_fn(query, key, value, mask=attn_mask)

        x = x.reshape(b, n_patches, self.in_features)
        out = self.out(x)
        return out, decode_cache


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention from a query stream into a static KV context.

    Q comes from the transformer's residual stream; K/V come from an external
    context tensor (e.g. Q-former output). No RoPE, no decode_cache: the KV
    context is treated as a static set of tokens at every layer / decode step.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        *,
        kv_features: int | None = None,
        use_bias: bool = False,
        attention_fn: Callable[..., torch.Tensor] = _torch_dot_product_attention,
        qk_norm: str = "rms",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.kv_features = kv_features if kv_features is not None else in_features
        self.head_dim = in_features // num_heads
        self.attention_fn = attention_fn

        if self.in_features % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.in_features}) must be divisible by 'num_heads' heads ({self.num_heads})."
            )

        self.q_proj = nn.Linear(self.in_features, self.in_features, bias=use_bias)
        self.k_proj = nn.Linear(self.kv_features, self.in_features, bias=use_bias)
        self.v_proj = nn.Linear(self.kv_features, self.in_features, bias=use_bias)
        self.out = nn.Linear(self.in_features, self.in_features, bias=use_bias)

        if qk_norm == "rms":
            self.query_ln = RMSNorm(self.head_dim)
            self.key_ln = RMSNorm(self.head_dim)
        else:
            self.query_ln = nn.Identity()
            self.key_ln = nn.Identity()

        self.per_dim_scale = PerDimScale(num_dims=self.head_dim)

    def forward(
        self,
        inputs_q: torch.Tensor,
        *,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n_q, _ = inputs_q.shape
        n_kv = kv.shape[1]

        if kv.dtype != inputs_q.dtype:
            kv = kv.to(dtype=inputs_q.dtype)

        query = self.q_proj(inputs_q).view(b, n_q, self.num_heads, self.head_dim)
        key = self.k_proj(kv).view(b, n_kv, self.num_heads, self.head_dim)
        value = self.v_proj(kv).view(b, n_kv, self.num_heads, self.head_dim)

        query = self.query_ln(query)
        key = self.key_ln(key)
        query = self.per_dim_scale(query)

        if kv_mask is None:
            attn_mask = torch.ones(
                b, 1, n_q, n_kv, dtype=torch.bool, device=inputs_q.device,
            )
        else:
            kv_valid = ~kv_mask  # True = valid
            attn_mask = kv_valid[:, None, None, :].expand(b, 1, n_q, n_kv)

        x = self.attention_fn(query, key, value, mask=attn_mask)
        x = x.reshape(b, n_q, self.in_features)
        return self.out(x)


class Transformer(nn.Module):
    """Classic Transformer used in Forecaster."""

    def __init__(
        self,
        model_dims: int,
        hidden_dims: int,
        num_heads: int,
        *,
        attention_norm: str,
        feedforward_norm: str,
        qk_norm: str,
        use_bias: bool,
        use_rotary_position_embeddings: bool,
        ff_activation: str,
        fuse_qkv: bool,
        use_cross_attn: bool = False,
        cross_attn_kv_dim: int = 0,
        use_cross_attn_gate: bool = True,
    ):
        super().__init__()

        if attention_norm == "rms":
            self.pre_attn_ln = RMSNorm(num_features=model_dims)
            self.post_attn_ln = RMSNorm(num_features=model_dims)
        else:
            raise ValueError(f"Layer norm: {attention_norm} not supported.")

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=model_dims,
            use_per_dim_scale=True,
            use_rotary_position_embeddings=use_rotary_position_embeddings,
            qk_norm=qk_norm,
            fuse_qkv=fuse_qkv,
        )

        if use_cross_attn:
            _cross_kv_dim = cross_attn_kv_dim or model_dims
            self.cross_attn_ln = RMSNorm(num_features=model_dims)
            self.cross_attn = MultiHeadCrossAttention(
                num_heads=num_heads,
                in_features=model_dims,
                kv_features=_cross_kv_dim,
                use_bias=use_bias,
                qk_norm=qk_norm,
            )
            if use_cross_attn_gate:
                # tanh(0)=0 → cross-attn contributes zero at init.
                self.cross_attn_gate = nn.Parameter(torch.zeros(1))
            else:
                self.cross_attn_gate = None
        else:
            self.cross_attn_ln = None
            self.cross_attn = None
            self.cross_attn_gate = None

        if feedforward_norm == "rms":
            self.pre_ff_ln = RMSNorm(num_features=model_dims)
            self.post_ff_ln = RMSNorm(num_features=model_dims)
        else:
            raise ValueError(f"Layer norm: {feedforward_norm} not supported.")

        self.ff0 = nn.Linear(
            in_features=model_dims,
            out_features=hidden_dims,
            bias=use_bias,
        )
        self.ff1 = nn.Linear(
            in_features=hidden_dims,
            out_features=model_dims,
            bias=use_bias,
        )
        if ff_activation == "relu":
            self.activation = nn.ReLU()
        elif ff_activation == "swish":
            self.activation = nn.SiLU()
        elif ff_activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation: {ff_activation} not supported.")

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
        decode_cache: DecodeCache | None = None,
        cross_kv: torch.Tensor | None = None,
        cross_kv_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecodeCache | None]:
        attn_output, decode_cache = self.attn(
            inputs_q=self.pre_attn_ln(input_embeddings),
            decode_cache=decode_cache,
            patch_mask=patch_mask,
        )
        attn_output = self.post_attn_ln(attn_output) + input_embeddings

        if self.cross_attn is not None and cross_kv is not None:
            cross_out = self.cross_attn(
                self.cross_attn_ln(attn_output),
                kv=cross_kv,
                kv_mask=cross_kv_mask,
            )
            if self.cross_attn_gate is not None:
                cross_out = torch.tanh(self.cross_attn_gate) * cross_out
            attn_output = attn_output + cross_out

        output_embeddings = (
            self.post_ff_ln(self.ff1(self.activation(self.ff0(self.pre_ff_ln(attn_output)))))
            + attn_output
        )
        return output_embeddings, decode_cache


class _QFormerBlock(nn.Module):
    """One Q-former block: self-attn over queries, cross-attn into source, FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ffn_mult: float = 4.0,
    ):
        super().__init__()
        self.self_attn_ln = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_ln_q = nn.LayerNorm(hidden_dim)
        self.cross_attn_ln_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        ffn_hidden = int(round(hidden_dim * ffn_mult))
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_norm = self.self_attn_ln(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        queries = queries + sa_out

        q_norm = self.cross_attn_ln_q(queries)
        kv_norm = self.cross_attn_ln_kv(kv)
        ca_out, _ = self.cross_attn(
            q_norm,
            kv_norm,
            kv_norm,
            key_padding_mask=kv_key_padding_mask,
            need_weights=False,
        )
        queries = queries + ca_out

        queries = queries + self.ffn(self.ffn_ln(queries))
        return queries


class QFormer(nn.Module):
    """BLIP-2-style Q-former: learned queries cross-attend to a source sequence.

    Args:
      in_dim: Source feature dim (e.g. ts encoder hidden, or LLM hidden).
      out_dim: Output (query) hidden dim. Should match the consumer module dim.
      num_query_tokens: Number of learned query tokens.
      num_heads: Attention heads inside the Q-former.
      num_layers: Number of stacked Q-former blocks.
      dropout: Attention/FFN dropout.
      ffn_mult: FFN expansion ratio.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_query_tokens: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        ffn_mult: float = 4.0,
    ):
        super().__init__()
        if num_query_tokens <= 0:
            raise ValueError(f"num_query_tokens must be positive, got {num_query_tokens}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_query_tokens = num_query_tokens

        self.input_proj = nn.Linear(in_dim, out_dim)
        self.input_ln = nn.LayerNorm(out_dim)

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, out_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _QFormerBlock(
                    hidden_dim=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_ln = nn.LayerNorm(out_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress `src` into `num_query_tokens` learned query vectors.

        Args:
          src: (B, T, in_dim) source token embeddings.
          src_key_padding_mask: (B, T) bool, True = padded position (ignored).

        Returns:
          (B, num_query_tokens, out_dim) query outputs.
        """
        if src.dim() != 3:
            raise ValueError(f"src must be (B, T, D), got shape {tuple(src.shape)}")
        if src.size(-1) != self.in_dim:
            raise ValueError(
                f"src last dim ({src.size(-1)}) does not match in_dim ({self.in_dim})"
            )

        kv = self.input_ln(self.input_proj(src))

        if src_key_padding_mask is not None:
            if src_key_padding_mask.shape != src.shape[:2]:
                raise ValueError(
                    "src_key_padding_mask shape must equal src.shape[:2]:"
                    f" {tuple(src_key_padding_mask.shape)} != {tuple(src.shape[:2])}"
                )
            kpm = src_key_padding_mask.to(dtype=torch.bool, device=kv.device)
            # If a row is fully padded, MHA would NaN. Flip one position to valid.
            all_padded = kpm.all(dim=1)
            if all_padded.any():
                kpm = kpm.clone()
                kpm[all_padded, 0] = False
        else:
            kpm = None

        queries = self.query_tokens.expand(src.size(0), -1, -1).to(dtype=kv.dtype)
        for block in self.blocks:
            queries = block(queries, kv, kv_key_padding_mask=kpm)

        return self.output_ln(queries)


class Aligner(nn.Module):
    """Aligns precomputed LLM / TS-encoder embeddings into Forecaster's cross-attn
    KV space, and predicts the forecast horizon.

    Two per-modality Q-formers compress the raw hidden-state sequences into a
    fixed number of query tokens; their concatenation is the static KV stream the
    Forecaster transformer cross-attends to. A small linear *prediction-length head*
    sits on the LLM Q-former's compressed output and regresses horizon.
    """

    def __init__(self, config: "InternS2PreviewTimeSeriesForecasterConfig"):
        super().__init__()
        qformer_kwargs = dict(
            out_dim=config.qformer_hidden_dim,
            num_query_tokens=config.qformer_num_query_tokens,
            num_heads=config.qformer_num_heads,
            num_layers=config.qformer_num_layers,
            dropout=config.qformer_dropout,
        )
        self.ts_qformer = QFormer(in_dim=config.d_ts_encoder, **qformer_kwargs)
        self.llm_qformer = QFormer(in_dim=config.d_llm, **qformer_kwargs)

        if config.use_horizon_head:
            # Prediction-length head: LayerNorm -> Linear -> SiLU -> Linear(.,1),
            # regressing horizon from the pooled LLM Q-former output.
            self.horizon_head = nn.Sequential(
                nn.LayerNorm(config.qformer_hidden_dim),
                nn.Linear(config.qformer_hidden_dim, config.qformer_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.qformer_hidden_dim, 1),
            )
        else:
            self.horizon_head = None

    def forward(
        self,
        llm_embedding_input: torch.Tensor,
        ts_encoder_embedding_input: torch.Tensor,
        llm_embedding_mask: torch.Tensor | None = None,
        ts_encoder_embedding_mask: torch.Tensor | None = None,
    ):
        """Compress both modalities into the Forecaster cross-attention KV stream.

        Args:
          llm_embedding_input: (B, T_llm, d_llm) precomputed LLM hidden states.
          ts_encoder_embedding_input: (B, T_ts, d_ts_encoder) precomputed TS-encoder
            hidden states.
          llm_embedding_mask / ts_encoder_embedding_mask: optional (B, T) bool masks,
            True = valid token. Default: all valid.

        Returns:
          ctx: (B, Q_ts + Q_llm, qformer_hidden_dim) cross-attention KV stream.
          llm_chunk: (B, Q_llm, qformer_hidden_dim) compressed LLM tokens.
        """
        # --- TS-encoder branch ---
        ts_param = next(self.ts_qformer.parameters())
        ts_hidden = ts_encoder_embedding_input.to(device=ts_param.device, dtype=ts_param.dtype)
        if ts_encoder_embedding_mask is None:
            ts_pad = torch.zeros(ts_hidden.shape[:2], dtype=torch.bool, device=ts_hidden.device)
        else:
            ts_pad = (~ts_encoder_embedding_mask.to(dtype=torch.bool)).to(device=ts_hidden.device)
        ts_chunk = self.ts_qformer(ts_hidden, src_key_padding_mask=ts_pad)

        # --- LLM branch ---
        llm_param = next(self.llm_qformer.parameters())
        llm_hidden = llm_embedding_input.to(device=llm_param.device, dtype=llm_param.dtype)
        if llm_embedding_mask is None:
            llm_pad = torch.zeros(llm_hidden.shape[:2], dtype=torch.bool, device=llm_hidden.device)
        else:
            llm_pad = (~llm_embedding_mask.to(dtype=torch.bool)).to(device=llm_hidden.device)
        llm_chunk = self.llm_qformer(llm_hidden, src_key_padding_mask=llm_pad)

        # --- Concatenate: ts tokens first, then llm tokens ---
        ctx = torch.cat([ts_chunk, llm_chunk], dim=1)
        return ctx, llm_chunk

    def predict_horizon(
        self,
        llm_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-sample forecast horizon in LINEAR step space from llm_qformer's output.

        Args:
          llm_chunk: (B, Q, qformer_hidden_dim) LLM Q-former output.

        Returns:
          (B,) float32 tensor of the predicted horizon (in steps) as float32. Linear (not log)
          so a small regression error stays a small ABSOLUTE step error — log1p/expm1 amplified
          errors at long horizons (e.g. ~0.05 in log ≈ 36 steps at H=720), hurting exact matching.
        """
        if self.horizon_head is None:
            raise RuntimeError("horizon_head is not enabled but predict_horizon was called")
        if llm_chunk.dim() != 3:
            raise ValueError(f"Expected llm_chunk with shape (B, Q, D), got {tuple(llm_chunk.shape)}")
        head_param = next(self.horizon_head.parameters())
        chunk = llm_chunk.to(device=head_param.device, dtype=head_param.dtype)
        pooled = chunk.mean(dim=1)
        return self.horizon_head(pooled).squeeze(-1).to(dtype=torch.float32)


class FutureCovariateEncoder(nn.Module):
    """Encode future covariates into patch-level KV tokens for Forecaster cross-attn.

    The first projection is shared across channels, so the module does not depend
    on a fixed number of covariate channels. For each future patch, learned query
    tokens pool all covariate channels into a small set of patch-local tokens.
    """

    def __init__(self, config: "InternS2PreviewTimeSeriesForecasterConfig"):
        super().__init__()
        self.patch_len = int(config.future_covariate_patch_len)
        self.queries_per_patch = int(config.future_covariate_queries_per_patch)
        self.hidden_dim = int(config.future_covariate_hidden_dim)
        self.output_dim = int(config.qformer_hidden_dim)
        self.max_horizon = int(config.future_covariate_max_horizon)
        self.max_patches = math.ceil(self.max_horizon / self.patch_len)
        self.max_tokens = self.max_patches * self.queries_per_patch

        if self.patch_len <= 0:
            raise ValueError(f"future_covariate_patch_len must be positive, got {self.patch_len}")
        if self.queries_per_patch <= 0:
            raise ValueError(
                f"future_covariate_queries_per_patch must be positive, got {self.queries_per_patch}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"future_covariate_hidden_dim must be positive, got {self.hidden_dim}")
        if self.max_horizon <= 0:
            raise ValueError(f"future_covariate_max_horizon must be positive, got {self.max_horizon}")

        self.channel_patch_proj = nn.Sequential(
            nn.LayerNorm(self.patch_len),
            nn.Linear(self.patch_len, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.patch_queries = nn.Parameter(torch.empty(self.queries_per_patch, self.hidden_dim))
        nn.init.trunc_normal_(self.patch_queries, std=0.02)
        self.channel_attn = nn.MultiheadAttention(
            self.hidden_dim,
            int(config.future_covariate_num_heads),
            dropout=float(config.future_covariate_dropout),
            batch_first=True,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        if int(config.future_covariate_num_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(config.future_covariate_num_heads),
                dim_feedforward=self.hidden_dim * 4,
                dropout=float(config.future_covariate_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_mixer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(config.future_covariate_num_layers),
            )
        else:
            self.token_mixer = None

        self.output_proj = (
            nn.Identity()
            if self.hidden_dim == self.output_dim
            else nn.Linear(self.hidden_dim, self.output_dim)
        )

    def _encode_one(self, covariates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        param = self.patch_queries
        covariates = covariates.to(device=param.device, dtype=param.dtype)
        if covariates.dim() != 2:
            raise ValueError(f"future covariates must be 2D (H, C), got {tuple(covariates.shape)}")

        horizon, channels = covariates.shape
        if horizon > self.max_horizon:
            raise ValueError(
                "Future covariate horizon exceeds configured maximum:"
                f" {horizon} > {self.max_horizon}"
            )

        num_patches = max(1, math.ceil(max(horizon, 1) / self.patch_len))
        num_tokens = num_patches * self.queries_per_patch
        if channels == 0 or horizon == 0:
            tokens = covariates.new_zeros(num_tokens, self.hidden_dim)
            mask = torch.ones(num_tokens, dtype=torch.bool, device=param.device)
            return tokens, mask

        pad_len = num_patches * self.patch_len - horizon
        if pad_len > 0:
            covariates = torch.cat(
                [covariates, covariates.new_zeros(pad_len, channels)],
                dim=0,
            )
        patches = covariates.reshape(num_patches, self.patch_len, channels).transpose(1, 2)
        channel_tokens = self.channel_patch_proj(patches)
        queries = self.patch_queries.unsqueeze(0).expand(num_patches, -1, -1)
        pooled, _ = self.channel_attn(
            queries,
            channel_tokens,
            channel_tokens,
            need_weights=False,
        )
        tokens = pooled.reshape(num_tokens, self.hidden_dim)
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=param.device)
        return tokens, mask

    def forward(self, future_covariates: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not future_covariates:
            raise ValueError("future_covariates must contain one tensor per sample")

        encoded = [self._encode_one(covariates) for covariates in future_covariates]
        max_tokens = max(tokens.shape[0] for tokens, _ in encoded)
        if max_tokens > self.max_tokens:
            raise ValueError(
                "Future covariate token count exceeds configured maximum:"
                f" {max_tokens} > {self.max_tokens}"
            )

        param = self.patch_queries
        batch_tokens = []
        batch_mask = []
        for tokens, mask in encoded:
            pad_tokens = max_tokens - tokens.shape[0]
            if pad_tokens > 0:
                tokens = torch.cat(
                    [tokens, tokens.new_zeros(pad_tokens, tokens.shape[-1])],
                    dim=0,
                )
                mask = torch.cat(
                    [mask, torch.ones(pad_tokens, dtype=torch.bool, device=param.device)],
                    dim=0,
                )
            batch_tokens.append(tokens)
            batch_mask.append(mask)

        tokens = torch.stack(batch_tokens, dim=0)
        mask = torch.stack(batch_mask, dim=0)
        tokens = tokens + self.position_embedding[:, :max_tokens].to(dtype=tokens.dtype, device=tokens.device)

        if self.token_mixer is not None:
            safe_mask = mask.clone()
            fully_masked = safe_mask.all(dim=1)
            if fully_masked.any():
                safe_mask[fully_masked, 0] = False
            tokens = self.token_mixer(tokens, src_key_padding_mask=safe_mask)
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)

        return self.output_proj(tokens), mask


class ForecasterBackbone(nn.Module):
    """Forecaster 2.5 with 200M parameters (cross-attention capable)."""

    # Architecture constants (Forecaster 2.5 200M)
    context_limit = 16384
    _quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _num_layers = 20

    # Residual block configs
    _tokenizer_kwargs = dict(input_dims=64, hidden_dims=1280, output_dims=1280, use_bias=True, activation="swish")
    _output_point_kwargs = dict(
        input_dims=1280, hidden_dims=1280, output_dims=1280, use_bias=False, activation="swish"
    )
    _output_quantile_kwargs = dict(
        input_dims=1280, hidden_dims=1280, output_dims=10240, use_bias=False, activation="swish"
    )

    # Transformer layer config
    _xf_kwargs = dict(
        model_dims=1280,
        hidden_dims=1280,
        num_heads=16,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        use_rotary_position_embeddings=True,
        ff_activation="swish",
        fuse_qkv=True,
    )

    def __init__(
        self,
        use_cross_attn: bool = False,
        cross_attn_kv_dim: int = 0,
        use_cross_attn_gate: bool = True,
    ):
        super().__init__()

        # Named constants.
        self.p = 32
        self.o = 128
        self.os = 1024
        self.m = self.o // self.p  # 4
        self.x = self._num_layers  # 20
        self.h = self._xf_kwargs["num_heads"]  # 16
        self.md = self._xf_kwargs["model_dims"]  # 1280
        self.hd = self.md // self.h  # 80
        self.q = len(self._quantiles) + 1  # 10
        self.aridx = 5

        self.use_cross_attn = bool(use_cross_attn)

        xf_kwargs = dict(self._xf_kwargs)
        if self.use_cross_attn:
            xf_kwargs["use_cross_attn"] = True
            xf_kwargs["cross_attn_kv_dim"] = int(cross_attn_kv_dim) or self.md
            xf_kwargs["use_cross_attn_gate"] = bool(use_cross_attn_gate)

        # Layers.
        self.tokenizer = ResidualBlock(**self._tokenizer_kwargs)
        self.stacked_xf = nn.ModuleList([Transformer(**xf_kwargs) for _ in range(self.x)])
        self.output_projection_point = ResidualBlock(**self._output_point_kwargs)
        self.output_projection_quantiles = ResidualBlock(**self._output_quantile_kwargs)

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        decode_caches: list | None = None,
        cross_kv: torch.Tensor | None = None,
        cross_kv_mask: torch.Tensor | None = None,
    ):
        """Forward pass — history-only path with cross-attention injection."""
        if cross_kv is not None and not getattr(self, "use_cross_attn", False):
            raise ValueError(
                "cross_kv was supplied but the Forecaster module was constructed with"
                " use_cross_attn=False; rebuild the module with use_cross_attn=True."
            )

        input_dtype = inputs.dtype
        model_dtype = next(self.parameters()).dtype
        if inputs.dtype != model_dtype:
            inputs = inputs.to(dtype=model_dtype)

        tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
        input_embeddings = self.tokenizer(tokenizer_inputs)

        if cross_kv is not None:
            cross_kv = cross_kv.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        if cross_kv_mask is not None:
            cross_kv_mask = cross_kv_mask.to(device=input_embeddings.device, dtype=torch.bool)

        if decode_caches is None:
            decode_caches = [None] * self.x
        new_decode_caches = []

        output_embeddings = input_embeddings
        token_masks = masks[..., -1]

        for i, layer in enumerate(self.stacked_xf):
            output_embeddings, new_cache = layer(
                output_embeddings,
                token_masks,
                decode_caches[i],
                cross_kv=cross_kv,
                cross_kv_mask=cross_kv_mask,
            )
            new_decode_caches.append(new_cache)

        output_ts = self.output_projection_point(output_embeddings)
        output_quantile_spread = self.output_projection_quantiles(output_embeddings)

        if output_ts.dtype != input_dtype:
            input_embeddings = input_embeddings.to(dtype=input_dtype)
            output_embeddings = output_embeddings.to(dtype=input_dtype)
            output_ts = output_ts.to(dtype=input_dtype)
            output_quantile_spread = output_quantile_spread.to(dtype=input_dtype)

        return (
            input_embeddings,
            output_embeddings,
            output_ts,
            output_quantile_spread,
        ), new_decode_caches


class InternS2PreviewTimeSeriesForecaster(BaseModel):
    """Standalone TimeOmni_v2 forecaster (cross-attention Forecaster head).

    Inputs (see :meth:`forward`): the raw multi-channel ``history``, plus the two
    precomputed embedding streams ``llm_embedding_input`` and
    ``ts_encoder_embedding_input``. The LLM and TS encoder themselves are NOT part
    of this model.
    """

    config_class = InternS2PreviewTimeSeriesForecasterConfig
    base_model_prefix = "interns2_preview_time_series_forecaster"
    main_input_name = "history"
    supports_gradient_checkpointing = False

    def __init__(self, config: InternS2PreviewTimeSeriesForecasterConfig):
        super().__init__(config)

        if config.future_covariate_injection not in ("none", "forecaster_cross_attn"):
            raise ValueError(
                "future_covariate_injection must be one of"
                f" ('none', 'forecaster_cross_attn'), got {config.future_covariate_injection!r}"
            )
        self.aligner = Aligner(config)
        self.future_covariate_encoder = (
            FutureCovariateEncoder(config)
            if config.future_covariate_injection == "forecaster_cross_attn"
            else None
        )
        self.forecaster = ForecasterBackbone(
            use_cross_attn=True,
            cross_attn_kv_dim=config.cross_attn_kv_dim,
            use_cross_attn_gate=config.use_cross_attn_gate,
        )

        # Normalize max_context / max_horizon to Forecaster patch multiples.
        if config.max_context % self.forecaster.p != 0:
            config.max_context = math.ceil(config.max_context / self.forecaster.p) * self.forecaster.p
        if config.max_horizon % self.forecaster.o != 0:
            config.max_horizon = math.ceil(config.max_horizon / self.forecaster.o) * self.forecaster.o
        if config.max_context + config.max_horizon > self.forecaster.context_limit:
            raise ValueError(
                "Context + horizon must be less than the context limit."
                f" {config.max_context} + {config.max_horizon} > {self.forecaster.context_limit}."
            )
        if config.use_continuous_quantile_head and (config.max_horizon > self.forecaster.os):
            raise ValueError(f"Continuous quantile head is not supported for horizons > {self.forecaster.os}.")

        self._horizon_max_length = int(config.horizon_max_length) or int(config.max_horizon)

        quantiles = [0.5] + list(self.forecaster._quantiles)
        self.register_buffer("forecaster_quantiles", torch.tensor(quantiles), persistent=False)

        self._hf_prefix = "time_series_forecaster."
        self._init_load_spec()

    @torch.no_grad()
    def init_weights(self):
        # Forecaster submodules initialize themselves in __init__.
        # Keep this no-op to avoid clobbering custom initialization.
        return

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]
    
    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        assert float8_handler is None

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype,
        )
        layer_mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype, cast_forward_inputs=False,
        )

        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None

        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        num_recompute_layers = int(len(self.forecaster.stacked_xf) * fsdp_config.vision_recompute_ratio)

        for layer_idx in tqdm(list(range(len(self.forecaster.stacked_xf))), desc="[TimeSeries Forecaster Fully Shard]"):
            layer = self.forecaster.stacked_xf[layer_idx]

            if layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(
                    layer,
                    preserve_rng_state=checkpoint_preserve_rng_state,
                    checkpoint_impl=CheckpointImpl.REENTRANT,
                )
                if self.compile_cfg:
                    layer.forward = torch.compile(layer.forward, fullgraph=True)

            self.forecaster.stacked_xf[layer_idx] = layer

            self._fully_shard(
                mesh=self.fsdp_mesh,
                mp_policy=layer_mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
                module=layer,
            )

        for layer_cur, layer_next in zip(
            self.forecaster.stacked_xf[:-1],
            self.forecaster.stacked_xf[1:],
        ):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

        self._fully_shard(
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self

    @staticmethod
    def _flip_timesfm_quantile_order(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(-1) <= 1:
            return tensor
        return torch.cat([tensor[..., :1], torch.flip(tensor[..., 1:], dims=(-1,))], dim=-1)

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if value <= 0:
            return multiple
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _flatten_channelwise(
        history: list[torch.Tensor],
        ctx: torch.Tensor,
        gt_ts: list[torch.Tensor] | None = None,
        ctx_mask: torch.Tensor | None = None,
        target_channel_idx: int | None = None,
    ):
        """Split each (T, C) history into C single-channel series; replicate the
        per-sample cross-attn context once per channel (Forecaster forecasts a single
        univariate series at a time)."""
        channel_counts = []
        flattened_inputs = []
        flattened_prefix = []
        flattened_prefix_mask = [] if ctx_mask is not None else None
        flattened_gt = [] if gt_ts is not None else None
        for sample_idx, ts in enumerate(history):
            if ts.dim() != 2:
                raise ValueError(
                    f"Each history series must be 2D (T, C), got shape {tuple(ts.shape)}"
                )
            channels = ts.shape[1]

            if gt_ts is not None:
                gt = gt_ts[sample_idx]
                if gt is None:
                    raise ValueError("Teacher-forced forecast requires gt_ts for every sample in the batch")
                if gt.dim() != 2:
                    raise ValueError(f"Each gt_ts must be 2D (T, C), got shape {tuple(gt.shape)}")
                if gt.shape[1] == channels:
                    channel_pairs = [(channel_idx, channel_idx) for channel_idx in range(channels)]
                elif gt.shape[1] == 1 and target_channel_idx is not None:
                    if target_channel_idx < 0 or target_channel_idx >= channels:
                        raise ValueError(
                            "forecast target channel index is out of range for input channels:"
                            f" idx={target_channel_idx}, channels={channels}"
                        )
                    channel_pairs = [(target_channel_idx, 0)]
                elif gt.shape[1] == 1 and channels == 5:
                    input_target_channel_idx = 3  # For FinMultiTime data, "channel_detail": ["Open", "High", "Low", "Close", "Volume"] and forecast "Close" price, so use channel index 3 as the target. This can be made configurable if needed.
                    channel_pairs = [(input_target_channel_idx, 0)]
                elif gt.shape[1] == 1 and channels == 45:
                    input_target_channel_idx = 0  # For NewElec data, "channel_detail": ["target", "weather_0", ..., "weather_43"]: channel 0 is the electricity load to forecast, channels 1..44 are weather covariates. Only channel 0 drives the TimesFM backbone; the full 45 channels still feed the TS encoder.
                    channel_pairs = [(input_target_channel_idx, 0)]
                else:
                    raise ValueError(
                        "Input/target channel mismatch is not supported:"
                        f" input shape {tuple(ts.shape)}, gt shape {tuple(gt.shape)}"
                    )
            else:
                channel_pairs = [(channel_idx, channel_idx) for channel_idx in range(channels)]

            channel_counts.append(len(channel_pairs))
            for input_channel_idx, gt_channel_idx in channel_pairs:
                flattened_inputs.append(ts[:, input_channel_idx])
                flattened_prefix.append(ctx[sample_idx])
                if flattened_prefix_mask is not None:
                    flattened_prefix_mask.append(ctx_mask[sample_idx])
                if flattened_gt is not None:
                    flattened_gt.append(gt[:, gt_channel_idx])
        return channel_counts, flattened_inputs, flattened_prefix, flattened_prefix_mask, flattened_gt

    @staticmethod
    def revin(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        reverse: bool = False,
    ):
        """Reversible instance normalization."""
        if len(mu.shape) == len(x.shape) - 1:
            mu = mu[..., None]
            sigma = sigma[..., None]
        elif len(mu.shape) == len(x.shape) - 2:
            mu = mu[..., None, None]
            sigma = sigma[..., None, None]

        if reverse:
            return x * sigma + mu
        else:
            return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)

    @staticmethod
    def update_running_stats(
        n: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Updates the running stats."""
        is_legit = torch.logical_not(mask)
        inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)

        inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
        inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
        inc_mu = inc_mu_numerator / inc_n_safe
        inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

        inc_var_numerator = torch.sum(
            ((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1
        )
        inc_var = inc_var_numerator / inc_n_safe
        inc_var = torch.where(inc_n == 0, 0.0, inc_var)
        inc_sigma = torch.sqrt(inc_var)

        new_n = n + inc_n
        new_n_safe = torch.where(new_n == 0, 1.0, new_n)

        new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
        new_mu = torch.where(new_n == 0, 0.0, new_mu)

        term1 = n * sigma.pow(2)
        term2 = inc_n * inc_sigma.pow(2)
        term3 = n * (mu - new_mu).pow(2)
        term4 = inc_n * (inc_mu - new_mu).pow(2)

        new_var = (term1 + term2 + term3 + term4) / new_n_safe
        new_var = torch.where(new_n == 0, 0.0, new_var)
        new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

        return (w := (new_n, new_mu, new_sigma), w)

    def forward(
        self,
        history: list[torch.Tensor],
        llm_embedding_input: torch.Tensor,
        ts_encoder_embedding_input: torch.Tensor,
        llm_embedding_mask: torch.Tensor | None = None,
        ts_encoder_embedding_mask: torch.Tensor | None = None,
        gt_ts: list[torch.Tensor] | None = None,
        future_covariates: list[torch.Tensor] | None = None,
        return_dict: bool | None = None,
    ):
        ctx, llm_chunk = self.aligner(
            llm_embedding_input,
            ts_encoder_embedding_input,
            llm_embedding_mask=llm_embedding_mask,
            ts_encoder_embedding_mask=ts_encoder_embedding_mask,
        )
        ctx_mask = None
        if self.future_covariate_encoder is not None:
            if future_covariates is None:
                raise ValueError(
                    "future_covariates must be provided when future_covariate_injection="
                    "'forecaster_cross_attn'"
                )
            if len(future_covariates) != ctx.shape[0]:
                raise ValueError(
                    "future_covariates batch size must match ctx batch size:"
                    f" {len(future_covariates)} != {ctx.shape[0]}"
                )
            future_cov_tokens, future_cov_mask = self.future_covariate_encoder(future_covariates)
            future_cov_tokens = future_cov_tokens.to(device=ctx.device, dtype=ctx.dtype)
            future_cov_mask = future_cov_mask.to(device=ctx.device)
            base_ctx_mask = torch.zeros(ctx.shape[:2], dtype=torch.bool, device=ctx.device)
            ctx = torch.cat([ctx, future_cov_tokens], dim=1)
            ctx_mask = torch.cat([base_ctx_mask, future_cov_mask], dim=1)

        horizon_loss = None
        if self.aligner.horizon_head is not None:
            pred_horizon = self.aligner.predict_horizon(llm_chunk)
            gt_lengths_tensor = torch.tensor(
                [int(gt.shape[0]) for gt in gt_ts],
                device=pred_horizon.device,
                dtype=torch.float32,
            )

            # Linear (step-space) regression: smooth_l1 directly on the GT length so the loss is
            # absolute-step error (== horizon_predict_mae), not relative error. No log1p — that
            # amplified long-horizon errors and hurt exact matching.
            horizon_loss = nn.functional.smooth_l1_loss(pred_horizon, gt_lengths_tensor)

        target_channel_idx = (
            int(self.config.future_covariate_target_channel_idx)
            if self.future_covariate_encoder is not None
            else None
        )
        channel_counts, flattened_inputs, flattened_prefix, flattened_prefix_mask, flattened_gt = self._flatten_channelwise(
            history,
            ctx,
            gt_ts,
            ctx_mask=ctx_mask,
            target_channel_idx=target_channel_idx,
        )

        flattened_prefix = [p.to(dtype=torch.float32) for p in flattened_prefix]

        patch_len = self.forecaster.p
        output_patch_len = self.forecaster.o
        quantile_patch_len = self.forecaster.os
        num_quantiles = self.forecaster.q

        max_input_len = max(int(ts.numel()) for ts in flattened_inputs)
        max_gt_len = max(int(ts.numel()) for ts in flattened_gt)
        input_batch_len = self._round_up_to_multiple(max_input_len, patch_len)
        gt_batch_len = self._round_up_to_multiple(max_gt_len, patch_len)
        total_seq_len = input_batch_len + gt_batch_len

        if total_seq_len > self.forecaster.context_limit:
            raise ValueError(
                "Teacher-forced forecast sequence exceeds TimesFM context limit:"
                f" {total_seq_len} > {self.timesfm_context_limit}"
            )
        
        forecaster_device = next(self.forecaster.parameters()).device

        full_inputs = []
        full_masks = []
        future_values = []
        future_valid_mask = []
        prefix_batch = [] if flattened_prefix is not None else None
        prefix_batch_mask = [] if flattened_prefix_mask is not None else None
        gt_lengths = []
        for series_idx, (hist_ts, future_ts) in enumerate(zip(flattened_inputs, flattened_gt)):
            hist_ts = hist_ts.to(device=forecaster_device).reshape(-1)
            future_ts = future_ts.to(device=forecaster_device).reshape(-1)
            gt_lengths.append(int(future_ts.numel()))

            # Truncate history to max_context (matching inference _preprocess)
            if hist_ts.numel() > input_batch_len:
                hist_ts = hist_ts[-input_batch_len:]

            input_pad_len = input_batch_len - int(hist_ts.numel())
            gt_pad_len = gt_batch_len - int(future_ts.numel())

            padded_hist = torch.cat(
                [
                    hist_ts.new_zeros(input_pad_len),
                    hist_ts,
                ],
                dim=0,
            )
            padded_future = torch.cat(
                [
                    future_ts,
                    future_ts.new_zeros(gt_pad_len),
                ],
                dim=0,
            )
            padded_future_mask = torch.cat(
                [
                    future_ts.new_ones(future_ts.numel()),
                    future_ts.new_zeros(gt_pad_len),
                ],
                dim=0,
            )
            full_inputs.append(torch.cat([padded_hist, padded_future], dim=0))
            future_values.append(padded_future)
            future_valid_mask.append(padded_future_mask)

            full_masks.append(
                torch.cat(
                    [
                        torch.ones(input_pad_len, device=forecaster_device, dtype=torch.bool),
                        torch.zeros(hist_ts.numel(), device=forecaster_device, dtype=torch.bool),
                        torch.zeros(future_ts.numel(), device=forecaster_device, dtype=torch.bool),
                        torch.ones(gt_pad_len, device=forecaster_device, dtype=torch.bool),
                    ],
                    dim=0,
                )
            )

            if prefix_batch is not None:
                prefix_batch.append(
                    flattened_prefix[series_idx]
                )
            if prefix_batch_mask is not None:
                prefix_batch_mask.append(
                    flattened_prefix_mask[series_idx]
                )

        full_inputs_t = torch.stack(full_inputs, dim=0)
        full_masks_t = torch.stack(full_masks, dim=0)

        # Compute is_positive from ORIGINAL history-only inputs before global normalization (matching inference)
        normalize_inputs = getattr(self.config, "normalize_inputs", False)
        if getattr(self.config, "infer_is_positive", False):
            history_inputs = full_inputs_t[:, :input_batch_len]
            history_masks = full_masks_t[:, :input_batch_len]
            valid_nonnegative = torch.where(
                history_masks,
                torch.ones(history_inputs.shape, device=forecaster_device, dtype=torch.bool),
                history_inputs >= 0,
            )
            is_positive = valid_nonnegative.all(dim=1)
        else:
            is_positive = None

        # Global normalization using history-only stats (matching inference _compiled_decode)
        global_mu = None
        global_sigma = None
        if normalize_inputs:
            history_part = full_inputs_t[:, :input_batch_len]
            global_mu = history_part.mean(dim=-1, keepdim=True)
            global_sigma = history_part.std(dim=-1, keepdim=True)
            full_inputs_t = self.revin(
                full_inputs_t, global_mu, global_sigma, reverse=False,
            )

        patched_inputs = full_inputs_t.reshape(full_inputs_t.size(0), -1, patch_len)
        patched_masks = full_masks_t.reshape(full_masks_t.size(0), -1, patch_len)

        running_n = full_inputs_t.new_zeros(full_inputs_t.size(0))
        running_mu = torch.zeros_like(running_n)
        running_sigma = torch.zeros_like(running_n)
        patch_mu = []
        patch_sigma = []
        for patch_idx in range(patched_inputs.size(1)):
            (running_n, running_mu, running_sigma), _ = self.update_running_stats(
                running_n,
                running_mu,
                running_sigma,
                patched_inputs[:, patch_idx, :],
                patched_masks[:, patch_idx, :],
            )
            patch_mu.append(running_mu)
            patch_sigma.append(running_sigma)

        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)
        normed_inputs = self.revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        prefix_batch_t = None
        prefix_batch_mask_t = None
        if prefix_batch is not None:
            prefix_batch_t = torch.stack(prefix_batch, dim=0).to(device=forecaster_device)
        if prefix_batch_mask is not None:
            prefix_batch_mask_t = torch.stack(prefix_batch_mask, dim=0).to(
                device=forecaster_device, dtype=torch.bool
            )

        (_, _, normed_outputs, normed_quantile_spread), _ = self.forecaster(
            normed_inputs,
            patched_masks,
            cross_kv=prefix_batch_t,
            cross_kv_mask=prefix_batch_mask_t,
        )

        if getattr(self.config, "force_flip_invariance", False):
            (_, _, flipped_normed_outputs, flipped_normed_quantile_spread), _ = self.forecaster(
                -normed_inputs,
                patched_masks,
                cross_kv=prefix_batch_t,
                cross_kv_mask=prefix_batch_mask_t,
            )
            # Reshape to 4D so _flip_timesfm_quantile_order flips only the quantile dim,
            # not the entire flattened output_patch_len * num_quantiles dimension.
            B = normed_outputs.size(0)
            flipped_normed_outputs = self._flip_timesfm_quantile_order(
                flipped_normed_outputs.reshape(B, -1, output_patch_len, num_quantiles)
            ).reshape(B, -1, output_patch_len * num_quantiles)
            flipped_normed_quantile_spread = self._flip_timesfm_quantile_order(
                flipped_normed_quantile_spread.reshape(B, -1, quantile_patch_len, num_quantiles)
            ).reshape(B, -1, quantile_patch_len * num_quantiles)
            normed_outputs = (normed_outputs - flipped_normed_outputs) / 2
            normed_quantile_spread = (normed_quantile_spread - flipped_normed_quantile_spread) / 2

        point_quantile_patches = self.revin(
            normed_outputs,
            context_mu,
            context_sigma,
            reverse=True,
        ).reshape(full_inputs_t.size(0), -1, output_patch_len, num_quantiles)

        quantile_spread_patches = self.revin(
            normed_quantile_spread,
            context_mu,
            context_sigma,
            reverse=True,
        ).reshape(full_inputs_t.size(0), -1, quantile_patch_len, num_quantiles)

        first_target_patch_idx = (input_batch_len // patch_len) - 1
        num_supervised_patches = gt_batch_len // patch_len
        pred_point_patches = point_quantile_patches[
            :, first_target_patch_idx:first_target_patch_idx + num_supervised_patches, :, 5
        ]
        pred_quantile_patches = point_quantile_patches[
            :, first_target_patch_idx:first_target_patch_idx + num_supervised_patches, :, :
        ].clone()
        if getattr(self.config, "use_continuous_quantile_head", False):
            max_spread_supervised_patches = (quantile_patch_len - output_patch_len) // patch_len + 1
            if num_supervised_patches > max_spread_supervised_patches:
                raise ValueError(
                    "Continuous quantile head spread from the last history patch cannot cover all supervised patches:"
                    f" {num_supervised_patches} > {max_spread_supervised_patches}"
                )

            history_quantile_spread = quantile_spread_patches[:, first_target_patch_idx, :, :]
            for rel_patch_idx in range(num_supervised_patches):
                spread_start = rel_patch_idx * patch_len
                spread_end = spread_start + output_patch_len
                for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
                    pred_quantile_patches[:, rel_patch_idx, :, quantile_index] = (
                        history_quantile_spread[:, spread_start:spread_end, quantile_index]
                        - history_quantile_spread[:, spread_start:spread_end, 5]
                        + pred_quantile_patches[:, rel_patch_idx, :, 5]
                    )

        if getattr(self.config, "fix_quantile_crossing", False):
            for quantile_index in [4, 3, 2, 1]:
                pred_quantile_patches[:, :, :, quantile_index] = torch.where(
                    pred_quantile_patches[:, :, :, quantile_index] < pred_quantile_patches[:, :, :, quantile_index + 1],
                    pred_quantile_patches[:, :, :, quantile_index],
                    pred_quantile_patches[:, :, :, quantile_index + 1],
                )
            for quantile_index in [6, 7, 8, 9]:
                pred_quantile_patches[:, :, :, quantile_index] = torch.where(
                    pred_quantile_patches[:, :, :, quantile_index] > pred_quantile_patches[:, :, :, quantile_index - 1],
                    pred_quantile_patches[:, :, :, quantile_index],
                    pred_quantile_patches[:, :, :, quantile_index - 1],
                )

        if is_positive is not None:
            if normalize_inputs:
                # In normalized space, 0 in original scale is -mu/sigma.
                # Clipping to -mu/sigma here is equivalent to clipping to 0
                # after global denorm, matching the inference path.
                zero_in_norm_3d = (-global_mu / global_sigma.clamp(min=1e-5)).unsqueeze(-1)
                zero_in_norm_4d = zero_in_norm_3d.unsqueeze(-1)
            else:
                zero_in_norm_3d = torch.zeros(1, device=forecaster_device)
                zero_in_norm_4d = torch.zeros(1, device=forecaster_device)
            pred_point_patches = torch.where(
                is_positive[:, None, None],
                torch.maximum(pred_point_patches, zero_in_norm_3d),
                pred_point_patches,
            )
            pred_quantile_patches = torch.where(
                is_positive[:, None, None, None],
                torch.maximum(pred_quantile_patches, zero_in_norm_4d),
                pred_quantile_patches,
            )

        point_loss_weight = float(getattr(self.config, "point_loss_weight", 1.0))
        quantile_loss_weight = float(getattr(self.config, "quantile_loss_weight", 1.0))
        quantiles = (
            self.forecaster_quantiles.to(device=forecaster_device, dtype=pred_quantile_patches.dtype)
            if quantile_loss_weight != 0.0
            else None
        )

        future_values = torch.stack(future_values, dim=0).to(dtype=pred_point_patches.dtype)
        future_valid_mask = torch.stack(future_valid_mask, dim=0).to(dtype=pred_point_patches.dtype)

        # Normalize GT values with the same global stats for loss in normalized space
        if normalize_inputs:
            future_values = self.revin(
                future_values, global_mu, global_sigma, reverse=False,
            )

        supervision_pad_len = output_patch_len - patch_len
        future_values_for_loss = torch.cat(
            [
                future_values,
                torch.zeros(
                    future_values.size(0),
                    supervision_pad_len,
                    device=forecaster_device,
                    dtype=future_values.dtype,
                ),
            ],
            dim=1,
        )
        future_mask_for_loss = torch.cat(
            [
                future_valid_mask,
                torch.zeros(
                    future_valid_mask.size(0),
                    supervision_pad_len,
                    device=forecaster_device,
                    dtype=future_valid_mask.dtype,
                ),
            ],
            dim=1,
        )

        label_patches = future_values_for_loss.unfold(dimension=1, size=output_patch_len, step=patch_len)
        loss_mask = future_mask_for_loss.unfold(dimension=1, size=output_patch_len, step=patch_len)

        valid_loss_count = loss_mask.sum()
        if valid_loss_count.item() <= 0:
            zero = torch.zeros((), device=forecaster_device, dtype=pred_point_patches.dtype)
            return {
                "loss": zero,
                "point_loss": zero,
                "quantile_loss": zero,
            }

        zero = torch.zeros((), device=forecaster_device, dtype=pred_point_patches.dtype)
        point_loss = (
            ((pred_point_patches - label_patches) ** 2 * loss_mask).sum() / valid_loss_count
            if point_loss_weight != 0.0
            else zero
        )

        if quantile_loss_weight != 0.0:
            quantile_errors = label_patches.unsqueeze(-1) - pred_quantile_patches
            quantile_q = quantiles.view(1, 1, 1, -1)
            quantile_loss_raw = torch.maximum(
                quantile_q * quantile_errors,
                (quantile_q - 1.0) * quantile_errors,
            )
            quantile_loss = (
                quantile_loss_raw * loss_mask.unsqueeze(-1)
            ).sum() / (valid_loss_count * num_quantiles).clamp_min(1.0)
        else:
            quantile_loss = zero

        horizon_loss_weight = float(getattr(self.config, "horizon_loss_weight", 0.0) or 0.0)

        point_loss = point_loss * point_loss_weight
        quantile_loss = quantile_loss * quantile_loss_weight
        if horizon_loss is not None and horizon_loss_weight != 0.0:
            horizon_loss = horizon_loss.to(dtype=point_loss.dtype) * horizon_loss_weight
        else:
            horizon_loss = zero

        result = {
            "point_loss": point_loss,
            "quantile_loss": quantile_loss,
            "horizon_loss": horizon_loss,
        }

        return result
