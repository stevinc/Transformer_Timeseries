from torch import nn
import torch
from models.temporal_fusion_t.linear_layer import LinearLayer

class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, 
                input_size,
                hidden_layer_size,
                dropout_rate=None,
                use_time_distributed=True,
                batch_first=False
                ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed

        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        self.activation_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.gated_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        
        return torch.mul(activation, gated), gated
