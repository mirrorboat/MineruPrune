from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
import torch

@dataclass
class LagrangianOutput(ModelOutput):
    loss: torch.Tensor = None
    expected_sparsity: torch.Tensor = None
    target_sparsity: torch.Tensor = None
    expected_head_sparsity: torch.Tensor = None
    target_head_sparsity: torch.Tensor = None
    expected_head_layer_sparsity: torch.Tensor = None
    target_head_layer_sparsity: torch.Tensor = None
    expected_intermediate_sparsity: torch.Tensor = None
    target_intermediate_sparsity: torch.Tensor = None
    expected_mlp_sparsity: torch.Tensor = None
    target_mlp_sparsity: torch.Tensor = None
    expected_layer_sparsity: torch.Tensor = None
    target_layer_sparsity: torch.Tensor = None
    expected_hidden_sparsity: torch.Tensor = None
    target_hidden_sparsity: torch.Tensor = None
    expected_qk_head_dim_sparsity: torch.Tensor = None
    target_qk_head_dim_sparsity: torch.Tensor = None
    expected_vo_head_dim_sparsity: torch.Tensor = None
    target_vo_head_dim_sparsity: torch.Tensor = None

@dataclass
class zsOutput(ModelOutput):
    hidden_z: torch.Tensor = None
    head_z: torch.Tensor = None
    head_layer_z: torch.Tensor = None
    qk_head_dim_z: torch.Tensor = None
    vo_head_dim_z: torch.Tensor = None
    intermediate_z: torch.Tensor = None
    mlp_z: torch.Tensor = None
    layer_z: torch.Tensor = None