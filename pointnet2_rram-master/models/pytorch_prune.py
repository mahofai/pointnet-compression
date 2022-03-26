from typing import OrderedDict
from unicodedata import decimal

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch.nn.utils import prune

from noise_layers import NoiseModule, NoiseConv, NoiseLinear
from binary_utils import BiConv2dLSR, BiLinearLSR, TriConv2d, TriLinear, NoiseBiConv2dLSR
from models.pointnet2_utils import PointNetSetAbstraction


def prune_model_l2_structured(model,c_prune_rate,n=2,dim=1):
  proportion = 1-(1/c_prune_rate)
  print("proportion:",proportion)
  for module in model.modules():
      if isinstance(module,PointNetSetAbstraction):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module, 'weight', proportion, n=n, dim=dim)
          prune.remove(module, 'weight')
          print("prune PointNetSetAbstraction")
      if isinstance(module,nn.Linear):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module, 'weight', proportion, n=n, dim=dim)
          prune.remove(module, 'weight')
          print("prune nn.Linear")
      if isinstance(module,nn.Conv1d):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module, 'weight', proportion, n=n, dim=dim)
          prune.remove(module, 'weight')
          print("prune nn.Conv1d")
      if isinstance(module,NoiseLinear):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module.linear, 'weight', proportion, n=n, dim=dim)
          prune.remove(module.linear, 'weight')
          print("prune NoiseLinear")


  return model

def prune_model_l1_structured(model,c_prune_rate):
  proportion = 1-(1/c_prune_rate)
  print("proportion:",proportion)
  for module in model.modules():
      if isinstance(module,PointNetSetAbstraction):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module, 'weight', proportion, n=1, dim=0)
          #prune.remove(module, 'weight')
          print("prune PointNetSetAbstraction")
      if isinstance(module,nn.Linear):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module, 'weight', proportion, n=1, dim=0)
          #prune.remove(module, 'weight')
          print("prune nn.Linear")
      if isinstance(module,nn.Conv1d):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module, 'weight', proportion, n=1, dim=0)
          #prune.remove(module, 'weight')
          print("prune nn.Conv1d")
      if isinstance(module,NoiseLinear):
          #prune.l1_unstructured(module, 'weight', 0.8)
          prune.ln_structured(module.linear, 'weight', proportion, n=1, dim=0)
          #prune.remove(module, 'weight')
          print("prune NoiseLinear")

  return model

def prune_model_global_unstructured(model,proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module_tups.append((module, 'weight'))
        if isinstance(module, nn.Conv1d):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model