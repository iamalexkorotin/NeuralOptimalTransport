import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MeanNegBilinear(nn.Module):
    """Bilinear (inner product) cost"""
    def __init__(self):
        super(MeanNegBilinear, self).__init__()
        
    def __call__(self, in_1, in_2):
        return -(in_1 * in_2).flatten(start_dim=1).mean()