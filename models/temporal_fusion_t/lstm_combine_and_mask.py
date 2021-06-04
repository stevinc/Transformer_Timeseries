from torch import nn
import torch
from models.temporal_fusion_t.gated_residual_network import GatedResidualNetwork


class LSTMCombineAndMask(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_layer_size, dropout_rate, use_time_distributed=False, batch_first=True):
        super(LSTMCombineAndMask, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout_rate = dropout_rate
        
        self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.hidden_layer_size, self.hidden_layer_size, self.num_inputs, self.dropout_rate, use_time_distributed=use_time_distributed, return_gate=True, batch_first=batch_first)

        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(GatedResidualNetwork(self.hidden_layer_size, self.hidden_layer_size, None, self.dropout_rate, use_time_distributed=use_time_distributed, return_gate=False, batch_first=batch_first))

        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedding, additional_context=None):
        # Add temporal features
        _, time_steps, embedding_dim, num_inputs = list(embedding.shape)
                
        flattened_embedding = torch.reshape(embedding,
                      [-1, time_steps, embedding_dim * num_inputs])

        expanded_static_context = additional_context.unsqueeze(1)

        if additional_context is not None:
            sparse_weights, static_gate = self.flattened_grn(flattened_embedding, expanded_static_context)
        else:
            sparse_weights = self.flattened_grn(flattened_embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            trans_emb_list.append(
              self.single_variable_grns[i](embedding[Ellipsis,i])
            )

        transformed_embedding = torch.stack(trans_emb_list, dim=-1)
        
        combined = transformed_embedding*sparse_weights
        
        temporal_ctx = combined.sum(dim=-1)

        return temporal_ctx, sparse_weights, static_gate
