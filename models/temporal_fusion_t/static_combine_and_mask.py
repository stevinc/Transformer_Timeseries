from torch import nn
import torch
from models.temporal_fusion_t.gated_residual_network import GatedResidualNetwork

class StaticCombineAndMask(nn.Module):
    def __init__(self, input_size, num_static, hidden_layer_size, dropout_rate, additional_context=None, use_time_distributed=False, batch_first=True):
        super(StaticCombineAndMask, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size =input_size
        self.num_static = num_static
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context

        if self.additional_context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_static*self.hidden_layer_size, self.hidden_layer_size, self.num_static, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_static*self.hidden_layer_size, self.hidden_layer_size, self.num_static, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first)


        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_static):
            self.single_variable_grns.append(GatedResidualNetwork(self.hidden_layer_size, self.hidden_layer_size, None, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding, additional_context=None):
        # Add temporal features
        _, num_static, _ = list(embedding.shape)
        flattened_embedding = torch.flatten(embedding, start_dim=1)
        if additional_context is not None:
            sparse_weights = self.flattened_grn(flattened_embedding, additional_context)
        else:
            sparse_weights = self.flattened_grn(flattened_embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_static):
            ##select slice of embedding belonging to a single input
            trans_emb_list.append(
              self.single_variable_grns[i](torch.flatten(embedding[:, i:i + 1, :], start_dim=1))
            )

        transformed_embedding = torch.stack(trans_emb_list, dim=1)

        combined = transformed_embedding*sparse_weights
        
        static_vec = combined.sum(dim=1)

        return static_vec, sparse_weights
