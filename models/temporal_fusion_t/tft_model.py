"""
Implementation of Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
"""

import math
import torch
import ipdb
import json
from torch import nn
from models.temporal_fusion_t.base import BaseModel
from models.temporal_fusion_t.add_and_norm import AddAndNorm
from models.temporal_fusion_t.gated_residual_network import GatedResidualNetwork
from models.temporal_fusion_t.gated_linear_unit import GLU
from models.temporal_fusion_t.linear_layer import LinearLayer
from models.temporal_fusion_t.lstm_combine_and_mask import LSTMCombineAndMask
from models.temporal_fusion_t.static_combine_and_mask import StaticCombineAndMask
from models.temporal_fusion_t.time_distributed import TimeDistributed
from models.temporal_fusion_t.interpretable_multi_head_attention import InterpretableMultiHeadAttention


class TFT(BaseModel):
    def __init__(self, raw_params):
        super(TFT, self).__init__()

        params = dict(raw_params)  # copy locally
        print(params)

        # Data parameters
        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.category_counts = json.loads(str(params['category_counts']))
        self.n_multiprocessing_workers = int(params['n_workers'])

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(
            str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(
            str(params['known_categorical_inputs']))

        # Network params
        self.quantiles = list(params['quantiles'])
        self.device = str(params['device'])
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['lr'])
        self.minibatch_size = int(params['batch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.num_stacks = int(params['stack_size'])
        self.num_heads = int(params['num_heads'])
        self.batch_first = True
        self.num_static = len(self._static_input_loc)
        self.num_inputs = len(self._known_regular_input_idx) + self.output_size
        self.num_inputs_decoder = len(self._known_regular_input_idx)

        # Serialisation options
        # self._temp_folder = os.path.join(params['model_folder'], 'tmp')
        # self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        # print('*** params ***')
        # for k in params:
        #   print('# {} = {}'.format(k, params[k]))

        #######
        time_steps = self.time_steps
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
          self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        print("num_categorical_variables")
        print(num_categorical_variables)
        self.embeddings = nn.ModuleList()
        for i in range(num_categorical_variables):
            embedding = nn.Embedding(self.category_counts[i], embedding_sizes[i])
            self.embeddings.append(embedding)

        self.static_input_layer = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.time_varying_embedding_layer = LinearLayer(input_size=1, size=self.hidden_layer_size, use_time_distributed=True, batch_first=self.batch_first)

        self.static_combine_and_mask = StaticCombineAndMask(
                input_size=self.input_size,
                num_static=self.num_static,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                additional_context=None,
                use_time_distributed=False,
                batch_first=self.batch_first)
        self.static_context_variable_selection_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_enrichment_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_state_h_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_state_c_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.historical_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.num_encoder_steps,
                num_inputs=self.num_inputs,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.future_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.num_encoder_steps,
                num_inputs=self.num_inputs_decoder,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, batch_first=self.batch_first)
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, batch_first=self.batch_first)

        self.lstm_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.lstm_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.static_enrichment_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                return_gate=True,
                batch_first=self.batch_first)

        self.self_attn_layer = InterpretableMultiHeadAttention(self.num_heads, self.hidden_layer_size, dropout_rate=self.dropout_rate)

        self.self_attention_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.self_attention_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.decoder_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                return_gate=False,
                batch_first=self.batch_first)

        self.final_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.final_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.output_layer = LinearLayer(
                input_size=self.hidden_layer_size,
                size=self.output_size * len(self.quantiles),
                use_time_distributed=True,
                batch_first=self.batch_first)

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.

        Args:
          self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1] # 192
        bs = self_attn_inputs.shape[:1][0] # [64]
        # create batch_size identity matrices
        mask = torch.cumsum(torch.eye(len_s).reshape((1, len_s, len_s)).repeat(bs, 1, 1), 1)
        return mask

    def get_tft_embeddings(self, all_inputs):
        time_steps = self.time_steps

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
          self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
              all_inputs[:, :, num_regular_variables:]

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[:,:, i].long())
            for i in range(num_categorical_variables)
        ]

        # Static inputs
        if self._static_input_loc:
            static_inputs = []
            for i in range(num_regular_variables):
                if i in self._static_input_loc:
                    reg_i = self.static_input_layer(regular_inputs[:, 0, i:i + 1])
                    static_inputs.append(reg_i)

            emb_inputs = []
            for i in range(num_categorical_variables):
                if i + num_regular_variables in self._static_input_loc:
                    emb_inputs.append(embedded_inputs[i][:, 0, :])

            static_inputs += emb_inputs
            static_inputs = torch.stack(static_inputs, dim=1)

        else:
            static_inputs = None

        # Targets
        obs_inputs = torch.stack([
            self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1].float())
            for i in self._input_obs_loc
        ], dim=-1)


        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx and i not in self._input_obs_loc:
                e = self.embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                e = self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, dim=-1)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = []
        for i in self._known_regular_input_idx:
            if i not in self._static_input_loc:
                known_regular_inputs.append(self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1].float()))

        known_categorical_inputs = []
        for i in self._known_categorical_input_idx:
            if i + num_regular_variables not in self._static_input_loc:
                known_categorical_inputs.append(embedded_inputs[i])

        known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, dim=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def forward(self, x):
        # Size definitions.
        time_steps = self.time_steps
        combined_input_size = self.input_size
        encoder_steps = self.num_encoder_steps
        all_inputs = x.to(self.device)

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs \
            = self.get_tft_embeddings(all_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = torch.cat([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], dim=-1)
        else:
            historical_inputs = torch.cat([
                  known_combined_layer[:, :encoder_steps, :],
                  obs_inputs[:, :encoder_steps, :]
              ], dim=-1)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        static_encoder, static_weights = self.static_combine_and_mask(static_inputs)
        static_context_variable_selection = self.static_context_variable_selection_grn(static_encoder)
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)
        historical_features, historical_flags, _ = self.historical_lstm_combine_and_mask(historical_inputs, static_context_variable_selection)
        future_features, future_flags, _ = self.future_lstm_combine_and_mask(future_inputs, static_context_variable_selection)

        history_lstm, (state_h, state_c) = self.lstm_encoder(historical_features, (static_context_state_h.unsqueeze(0), static_context_state_c.unsqueeze(0)))
        future_lstm, _ = self.lstm_decoder(future_features, (state_h, state_c))

        lstm_layer = torch.cat([history_lstm, future_lstm], dim=1)
        # Apply gated skip connection
        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        lstm_layer, _ = self.lstm_glu(lstm_layer)
        temporal_feature_layer = self.lstm_glu_add_and_norm(lstm_layer, input_embeddings)

        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)
        enriched, _ = self.static_enrichment_grn(temporal_feature_layer, expanded_static_context)

        # Decoder self attention
        mask = self.get_decoder_mask(enriched)
        x, self_att = self.self_attn_layer(enriched, enriched, enriched, mask)#, attn_mask=mask.repeat(self.num_heads, 1, 1))

        x, _ = self.self_attention_glu(x)
        x = self.self_attention_glu_add_and_norm(x, enriched)

        # Nonlinear processing on outputs
        decoder = self.decoder_grn(x)
        # Final skip connection
        decoder, _ = self.final_glu(decoder)
        transformer_layer = self.final_glu_add_and_norm(decoder, temporal_feature_layer)
        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        outputs = self.output_layer(transformer_layer[:, self.num_encoder_steps:, :])
        return outputs, all_inputs, attention_components
