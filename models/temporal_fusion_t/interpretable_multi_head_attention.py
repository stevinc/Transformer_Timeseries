from torch import nn
import torch
from models.temporal_fusion_t.scaled_dot_product_attention import ScaledDotProductAttention

class InterpretableMultiHeadAttention(nn.Module):
    """Defines interpretable multi-head attention layer.

    Attributes:
      n_head: Number of heads
      d_k: Key/query dimensionality per head
      d_v: Value dimensionality
      dropout: Dropout rate to apply
      qs_layers: List of queries across heads
      ks_layers: List of keys across heads
      vs_layers: List of values across heads
      attention: Scaled dot product attention layer
      w_o: Output weight matrix to project internal state to the original TFT
        state size
    """

    def __init__(self, n_head, d_model, dropout_rate):
        """Initialises layer.

        Args:
          n_head: Number of heads
          d_model: TFT state dimensionality
          dropout: Dropout discard rate
        """
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = nn.Dropout(dropout_rate)

        self.qs_layers = nn.ModuleList()
        self.ks_layers = nn.ModuleList()
        self.vs_layers = nn.ModuleList()

        # Use same value layer to facilitate interp
        vs_layer = nn.Linear(d_model, d_v, bias=False)
        qs_layer = nn.Linear(d_model, d_k, bias=False)
        ks_layer = nn.Linear(d_model, d_k, bias=False)

        for _ in range(n_head):
            self.qs_layers.append(qs_layer)
            self.ks_layers.append(ks_layer)
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention = ScaledDotProductAttention()
        self.w_o = nn.Linear(self.d_k, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        """Applies interpretable multihead attention.

          Using T to denote the number of time steps fed into the transformer.

          Args:
            q: Query tensor of shape=(?, T, d_model)
            k: Key of shape=(?, T, d_model)
            v: Values of shape=(?, T, d_model)
            mask: Masking if required with shape=(?, T, T)

          Returns:
            Tuple of (layer outputs, attention weights)
          """
        n_head = self.n_head
        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = torch.stack(heads) if n_head > 1 else heads[0]
        attn = torch.stack(attns)

        outputs = torch.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = self.dropout(outputs)  # output dropout

        return outputs, attn
