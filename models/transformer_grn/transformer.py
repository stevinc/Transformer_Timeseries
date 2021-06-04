import torch
import torch.nn as nn

from models.transformer_grn.encoder import Encoder
from models.transformer_grn.decoder import Decoder
from models.transformer.utils import generate_original_PE, generate_regular_PE
from models.temporal_fusion_t.linear_layer import LinearLayer


class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    """

    def __init__(self, cnf: dict):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        d_model = cnf["d_model"]
        q = cnf["q"]
        v = cnf["v"]
        h = cnf["h"]
        N = cnf["N"]
        attention_size = cnf["attention_size"]
        dropout = cnf["dropout"]
        pe = cnf["pe"]
        chunk_mode = cnf["chunk_mode"]
        d_input = cnf["d_input"]
        d_output = cnf["d_output"]
        self.time_steps = cnf["num_encoder_steps"]
        self.static_vars = cnf['static_input_loc']
        self.regular_vars = cnf['known_regular_inputs'] + cnf['input_obs_loc']

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding_categorical = nn.ModuleList()
        for i in range(len(self.static_vars)):
            embedding = nn.Embedding(cnf['category_counts'][i], d_model)
            self._embedding_categorical.append(embedding)

        self._time_varying_embedding_layer = LinearLayer(input_size=len(self.regular_vars), size=d_model,
                                                         use_time_distributed=True, batch_first=True)

        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        else:
            self._generate_PE = None

        self.name = 'transformer'

    def split_features(self, x):
        x_static = torch.stack([
            self._embedding_categorical[i](x[..., ix].long())
            for i, ix in enumerate(self.static_vars)
        ], dim=-1)

        x_static = x_static[:, 0:1, :].squeeze(-1)
        x_input = self._time_varying_embedding_layer(x[..., self.regular_vars])

        return x_input, x_static

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        x = xy[:, :self.time_steps]
        y = xy[:, self.time_steps:]

        # Shift tensor add start token
        pad = torch.ones((y.shape[0], 1, y.shape[2])).to(y.device)
        y = torch.cat((pad, y), dim=1)[:, :-1, :]

        x_input, x_static = self.split_features(x)
        y_input, y_static = self.split_features(y)

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(x_input.shape[1], self._d_model)
            positional_encoding = positional_encoding.to(x_input.device)
            x_input.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding_x = layer(x_input, context=x_static)

        # Decoding stack
        decoding = y_input

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(y.shape[1], self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding_x, context=y_static)

        # Output module
        output = self._linear(decoding)
        return output
