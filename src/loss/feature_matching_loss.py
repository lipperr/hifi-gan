import torch
from torch import Tensor
import torch.nn as nn

class FeatureMatchingLoss(nn.Module):

    def __init__(self):

        pass
    def forward(
        self, gen_pred, melspectrogam, **batch
    ) -> Tensor:
        
        pass
        # return {"loss": loss}