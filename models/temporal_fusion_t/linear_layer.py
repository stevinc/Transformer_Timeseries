from torch import nn
import torch
from models.temporal_fusion_t.time_distributed import TimeDistributed

class LinearLayer(nn.Module):
    def __init__(self,
                input_size,
                size,
                use_time_distributed=True,
                batch_first=False):
        super(LinearLayer, self).__init__()

        self.use_time_distributed=use_time_distributed
        self.input_size=input_size
        self.size=size
        if use_time_distributed:
            self.layer = TimeDistributed(nn.Linear(input_size, size), batch_first=batch_first)
        else:
            self.layer = nn.Linear(input_size, size)
      
    def forward(self, x):
        return self.layer(x)
