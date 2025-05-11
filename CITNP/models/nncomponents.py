import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(dim_in, dim_hid, dim_out, depth):
    assert depth >= 2, "MLP depth must be at least 2."
    layers = [nn.Linear(dim_in, dim_hid), nn.GELU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(dim_hid, dim_hid), nn.GELU()]
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)


class BasePredictor(nn.Module):
    def __init__(
        self,
        d_model,
        depth,
        output_heads,
        num_outcome_nodes=1,
        num_mixture_components=1,
    ):
        super().__init__()
        self.output_heads = output_heads
        self.num_outcome_nodes = num_outcome_nodes
        self.num_mixture_components = num_mixture_components
        self.num_heads = len(output_heads)

        self.pred = build_mlp(
            d_model,
            d_model,
            dim_out=d_model,
            depth=depth,
        )

        total_output_dim = self.num_heads * num_outcome_nodes * num_mixture_components
        self.output_layer = nn.Linear(d_model, total_output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        self.output_layer.weight.data *= 0.01

    def forward(self, x):
        raise NotImplementedError("Use a subclass like Predictor or MoGPredictor.")


class Predictor(BasePredictor):
    """
    A simple feedforward neural network with a linear layer followed by a GELU activation.
    """

    def __init__(self, **kwargs):
        super().__init__(
            output_heads=("mean", "std"), num_mixture_components=1, **kwargs
        )

    def forward(self, x):
        act_out = F.gelu(self.pred(x))  # [B, N, D]
        D = self.num_outcome_nodes

        out = self.output_layer(act_out)  # [B, N, 2 * D]
        out = einops.rearrange(out, "... n (h d) -> h ... n d", h=self.num_heads, d=D)
        mean, pre_std = out[0], out[1]
        std = F.softplus(pre_std)  # [B, N, D]
        return mean, std


class MoGPredictor(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(output_heads=("mean", "std", "weights"), **kwargs)

    def forward(self, x):
        act_out = F.gelu(self.pred(x))  # [B, N, D]
        D = self.num_outcome_nodes
        K = self.num_mixture_components

        out = self.output_layer(act_out)  # [B, N, 3 * D * K]
        out = einops.rearrange(
            out, "b n (h d k) -> h b n d k", h=self.num_heads, d=D, k=K
        )

        mean, pre_std, logits = out[0], out[1], out[2]
        std = F.softplus(pre_std)  # [B, N, D, K]
        weights = F.softmax(logits, dim=-1)  # [B, N, D, K]
        return mean, std, weights


class FlashMHCA(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        batch_first: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        assert batch_first, "FlashMHCA only supports batch_first=True for now"
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        factory_kwargs = {"device": device, "dtype": dtype}

        self.layernormQ = nn.LayerNorm(d_model, **factory_kwargs)
        self.layernormK = nn.LayerNorm(d_model, **factory_kwargs)
        self.layernormV = nn.LayerNorm(d_model, **factory_kwargs)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        -----
        - query, key, value: shape [batch_size, seq_len, d_model]

        Returns:
        --------
        - output: shape [batch_size, seq_len, d_model]
        """
        B, T_q, _ = query.shape  # batch size, sequence length, embedding
        _, T_k, _ = key.shape  # batch size, sequence length, embedding

        query = self.layernormQ(query)
        key = self.layernormK(key)
        value = self.layernormV(value)

        q = (
            self.q_proj(query).view(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        )  # [B, H, T, D]
        k = self.k_proj(key).view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )  # [B, H, T, D]

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        )
        return self.out_proj(attn_output)


class MHCA(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        batch_first: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.layernormQ = nn.LayerNorm(d_model)
        self.layernormK = nn.LayerNorm(d_model)
        self.layernormV = nn.LayerNorm(d_model)

        self.z_rep_merger = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=batch_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        -----
        - query: torch.Tensor, shape [batch_size, num_samples, d_model]
        - key: torch.Tensor, shape [batch_size, num_samples, d_model]
        - value: torch.Tensor, shape [batch_size, num_samples, d_model]

        Returns:
        --------
        - torch.Tensor, shape [batch_size, num_samples, d_model]
        """
        # shape batch_size, num_samples, d_model]
        query = self.layernormQ(query)
        key = self.layernormK(key)
        value = self.layernormV(value)
        # shape [batch_size, num_samples, d_model]
        output, _ = self.z_rep_merger(query, key, value)
        return output


class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(2 * x)


class ResidualEncoderBlock(nn.Module):
    """
    A modular encoder block with a residual connection.
    Consists of:
      1) Two Linear layers with ReLU activation
      2) Optional projection if input_dim != output_dim
      3) Xavier initialization for all linear layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim, zero_init=True):
        super(ResidualEncoderBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # If input_dim != output_dim, learn a projection for the residual
        self.projection = None
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)

        self.activation = F.gelu

        self.zero_init = zero_init

        self.init_weights()

    def init_weights(self):
        """
        Initializes linear layers with Xavier initialization.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        if self.zero_init:
            self.fc2.weight.data[:] = 0

    def forward(self, x):
        # Store the original input for the residual connection
        residual = x

        # Pass through MLP
        out = self.activation(self.fc1(x))
        out = self.fc2(out)

        # If dimension changed, project residual to match out
        if self.projection is not None:
            residual = self.projection(residual)

        # Add the residual connection
        out = out + residual
        return out


def build_residual_network(
    num_blocks, input_dim, hidden_dim, output_dim, device=None, dtype=None
):
    """
    Builds a sequential network composed of `num_blocks` ResidualEncoderBlocks.

    - The first block uses (input_dim -> hidden_dim -> output_dim).
    - Subsequent blocks use (output_dim -> hidden_dim -> output_dim).

    Returns:
        nn.Sequential: A stack of `ResidualEncoderBlock`s.
    """
    layers = []
    for i in range(num_blocks):
        if i == 0:
            # First block: from input_dim to output_dim
            in_dim = input_dim
        else:
            # Subsequent blocks: from the output_dim of the previous block
            in_dim = output_dim

        block = ResidualEncoderBlock(in_dim, hidden_dim, output_dim)
        layers.append(block)

    model = nn.Sequential(*layers)

    model = model.to(device=device, dtype=dtype)

    return model
