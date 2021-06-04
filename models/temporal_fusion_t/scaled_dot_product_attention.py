from torch import nn
import torch

class ScaledDotProductAttention(nn.Module):
    """Defines scaled dot product attention layer.

      Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
          softmax by default)
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attn_dropout)
        self.activation = nn.Softmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
          q: Queries
          k: Keys
          v: Values
          mask: Masking if required -- sets softmax to very large value

        Returns:
          Tuple of (layer outputs, attention weights)
        """
        attn = torch.bmm(q,k.permute(0,2,1)) # shape=(batch, q, k)
        if mask is not None:
            attn = attn.masked_fill(mask.bool().to(self.device), -1e9)

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn,v)
        return output, attn
