import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CITNP.networks.attention_layers import MultiHeadSelfAttentionLayer


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

        total_output_dim = (
            self.num_heads * num_outcome_nodes * num_mixture_components
        )
        self.output_layer = nn.Linear(d_model, total_output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        self.output_layer.weight.data *= 0.01

    def forward(self, x):
        raise NotImplementedError(
            "Use a subclass like Predictor or MoGPredictor."
        )


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
        out = einops.rearrange(
            out, "... n (h d) -> h ... n d", h=self.num_heads, d=D
        )
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


class JointMoGPredictor(nn.Module):
    """Joint prediction head: per query row, a K-component mixture of
    D-variate Gaussians with low-rank-plus-diagonal covariances,

        p(x) = sum_k w_k N(x; mu_k, F_k F_k^T + diag(v_k)),

    read from the tokens of all D variables of the row. The idea follows the
    decoder head `GMMNetwork` of ACTIVA (Sauter et al.,
    github.com/sa-and/ACTIVA, models.py); this is an independent PyTorch
    implementation without its latent variable.

    Per query row, with x [D, d_model] the row's variable tokens:
    1. optional mixing: `node_attention_layers` self-attention layers over the
       D tokens of the row (rows never attend to each other; no positional
       information, so the head stays node-permutation equivariant). In the
       r1, layerwise and extra_node encodings every encoder block ends with
       attention across variables, so 0 (the default) suffices. In reg_axis
       and reg_axis_full every block ends with a per-variable read of the
       context, so what a variable learned in its last read has not reached
       the other variables; their joint configs should use 1, since the
       covariance needs the tokens of two variables to agree.
    2. trunk: hidden = gelu(MLP(x)) per variable (as `BasePredictor.pred`);
    3. per variable j and component k: mu_kj (Linear(d_model, K)), the factor
       row F_kj [R] (Linear(d_model, K * R)) and
       v_kj = softplus(Linear(d_model, K)) + minimum_variance;
    4. mixture logits: Linear(d_model, K) of the mean of hidden over the D
       variables.
    No parameter depends on D, so the head is node-permutation equivariant
    and works for any number of variables.

    Initial scale: the output layers are initialised like
    `BasePredictor._init_weights` (Kaiming, zero bias, weights x 0.01), so at
    initialisation the raw diagonal output is about 0 and the diagonal is
    about softplus(0) = 0.693 (standard deviation about 0.83 on standardised
    data), not `minimum_variance`, which is only a floor. Means and factor
    entries start near 0 and the mixture weights near uniform.

    The parameters output for an intervened variable are auxiliary: the loss
    never uses them and they are not a prediction for that variable.

    Args:
    -----
    - d_model (int): token width.
    - depth (int): depth of the trunk MLP (>= 2).
    - num_mixture_components (int): K >= 1.
    - covariance_rank (int): R >= 1, columns of each factor F_k.
    - minimum_variance (float): floor added to every diagonal entry v_kj.
    - node_attention_layers (int): self-attention layers over the variables
      of a row before the trunk (>= 0).
    - nhead, dim_feedforward (int): settings of those layers, as for the
      encoder's node attention (pre-norm, GELU, no dropout, softmax
      attention).
    """

    output_heads = (
        "component_mean",
        "covariance_factor",
        "covariance_diagonal",
        "mixture_logits",
    )

    def __init__(
        self,
        d_model: int,
        depth: int,
        num_mixture_components: int,
        covariance_rank: int,
        minimum_variance: float,
        node_attention_layers: int,
        nhead: int,
        dim_feedforward: int,
    ):
        super().__init__()
        if num_mixture_components < 1:
            raise ValueError("num_mixture_components must be >= 1")
        if covariance_rank < 1:
            raise ValueError("covariance_rank must be >= 1")
        if not minimum_variance > 0:
            raise ValueError("minimum_variance must be > 0")
        if node_attention_layers < 0:
            raise ValueError("node_attention_layers must be >= 0")
        self.num_mixture_components = num_mixture_components
        self.covariance_rank = covariance_rank
        self.minimum_variance = float(minimum_variance)
        self.node_attention = nn.ModuleList(
            MultiHeadSelfAttentionLayer(
                embed_dim=d_model,
                num_heads=nhead,
                head_dim=d_model // nhead,
                feedforward_dim=dim_feedforward,
                p_dropout=0.0,
                norm_first=True,
                activation=nn.GELU(),
                linear=False,
            )
            for _ in range(node_attention_layers)
        )
        self.pred = build_mlp(d_model, d_model, dim_out=d_model, depth=depth)
        K, R = num_mixture_components, covariance_rank
        self.mean_layer = nn.Linear(d_model, K)
        self.factor_layer = nn.Linear(d_model, K * R)
        self.diagonal_layer = nn.Linear(d_model, K)
        self.logits_layer = nn.Linear(d_model, K)
        self._init_weights()

    def output_layers(self) -> tuple[nn.Linear, ...]:
        return (
            self.mean_layer,
            self.factor_layer,
            self.diagonal_layer,
            self.logits_layer,
        )

    def _init_weights(self):
        """As `BasePredictor._init_weights`, for each output layer."""
        for layer in self.output_layers():
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer.weight.data *= 0.01

    def forward(self, node_encoding: torch.Tensor):
        """
        Args:
        -----
        - node_encoding (torch.Tensor): [B, Nq, D, d_model], the tokens of
            all variables of every query row (any leading dimensions work).

        Returns:
        --------
        - component_mean: [B, Nq, K, D].
        - covariance_factor: [B, Nq, K, D, R].
        - covariance_diagonal: [B, Nq, K, D], >= minimum_variance.
        - mixture_logits: [B, Nq, K].
        """
        x = node_encoding
        if len(self.node_attention) > 0:
            # [B * Nq, D, d_model]: each row's variables attend to each other
            rows = x.reshape(-1, *x.shape[-2:])
            for layer in self.node_attention:
                rows = layer(rows, mask=None)
            x = rows.reshape(x.shape)
        K, R = self.num_mixture_components, self.covariance_rank
        hidden = F.gelu(self.pred(x))  # [B, Nq, D, d_model]
        # [B, Nq, D, K] -> [B, Nq, K, D]
        component_mean = self.mean_layer(hidden).transpose(-1, -2)
        # [B, Nq, D, K * R] -> [B, Nq, D, K, R] -> [B, Nq, K, D, R]
        factor = self.factor_layer(hidden).unflatten(-1, (K, R))
        covariance_factor = factor.transpose(-3, -2)
        diagonal = F.softplus(self.diagonal_layer(hidden)) + self.minimum_variance
        covariance_diagonal = diagonal.transpose(-1, -2)  # [B, Nq, K, D]
        mixture_logits = self.logits_layer(hidden.mean(dim=-2))  # [B, Nq, K]
        return (
            component_mean,
            covariance_factor,
            covariance_diagonal,
            mixture_logits,
        )


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
            self.q_proj(query)
            .view(B, T_q, self.nhead, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, T, D]
        k = (
            self.k_proj(key)
            .view(B, T_k, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(B, T_k, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

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
    def __init__(self, omega=2.0):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


class ResidualEncoderBlock(nn.Module):
    """
    A modular encoder block with a residual connection.
    Consists of:
      1) Two Linear layers with ReLU activation
      2) Optional projection if input_dim != output_dim
      3) Xavier initialization for all linear layers
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        zero_init=True,
        random_activation=False,
    ):
        super(ResidualEncoderBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # If input_dim != output_dim, learn a projection for the residual
        self.projection = None
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)

        if random_activation:
            # Choose from multiple activations
            choice = np.random.choice(
                ["relu", "leaky_relu", "gelu", "sine", "swish"]
            )

            if choice == "relu":
                self.activation = F.relu
            elif choice == "leaky_relu":
                # Random leaky slope
                alpha = np.random.uniform(0.01, 0.3)
                self.activation = lambda x: F.leaky_relu(x, negative_slope=alpha)
            elif choice == "gelu":
                self.activation = F.gelu
            elif choice == "sine":
                omega_init = np.random.uniform(0.5, 10.0)
                self.activation = SineActivation(omega=omega_init)
            elif choice == "swish":
                self.activation = lambda x: x * torch.sigmoid(x)
        else:
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
