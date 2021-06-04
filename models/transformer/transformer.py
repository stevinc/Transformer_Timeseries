import torch
import torch.nn as nn

from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.transformer.utils import generate_original_PE, generate_regular_PE


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

        self._embedding_input = nn.Linear(d_input, d_model)
        self._embedding_output = nn.Linear(d_input, d_model)

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

        # Embeddin module
        encoding_x = self._embedding_input(x)
        encoding_y = self._embedding_output(y)

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(x.shape[1], self._d_model)
            positional_encoding = positional_encoding.to(encoding_x.device)
            encoding_x.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding_x = layer(encoding_x)

        # Decoding stack
        decoding = encoding_y

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(y.shape[1], self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding_x)

        # Output module
        output = self._linear(decoding)
        return output
