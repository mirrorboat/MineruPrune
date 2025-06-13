import math
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from argparse import Namespace as NS
from typing import Any, List

from llava.model.my_output_utils import LagrangianOutput, zsOutput
import torch.utils.checkpoint as checkpoint
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str,
                 eval_target_model: bool=True) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_sparsity=target_sparsity

        self.droprate_init = 0.5
        self.temperature = 2./3.
        self.magical_number = 0.8
        self.device = device
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1] # the full size of each unit
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model
        self.use_samlpe = True
        
    def param_init_fn(self, module):
        """ Initialize the parameters for masking variables. """
        mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        mean = 5
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)
        
    def initialize_mask(self, mask_shape: List):
        """ Initialize the parameters for masking variables. """
        z_loga = nn.Parameter(torch.ones(*mask_shape, device=self.device))
        self.param_init_fn(z_loga)
        return z_loga

    def cdf_qz(self, z_loga: torch.Tensor = None):
        """Implements the CDF of the 'stretched' concrete distribution"""
        if z_loga is None:
            z_loga = self.z_loga
        xn = (0 - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - z_loga).clamp(min=epsilon, max=1 - epsilon)
    
    def get_eps(self, size: List):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps) # is it a must?
        return eps
    
    def quantile_concrete(self, eps: torch.Tensor):
        y = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + self.z_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a
    
    def sample_z(self):
        eps = self.get_eps(torch.FloatTensor(*self.z_loga.shape)).to(self.z_loga.device)
        # eps = torch.rand_like(self.z_loga).clamp(min=epsilon, max=1 - epsilon)
        z = self.quantile_concrete(eps)
        z = F.hardtanh(z, min_val=0, max_val=1).reshape(*self.mask_output_shape)
        return z
    
    def _deterministic_z(self, z_loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        if self.target_mask_size is None or not self.eval_target_model:
            print("WARNING eval_target_model is False")
            expected_score = 1 - self.cdf_qz(z_loga)
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.nelement() - expected_num_nonzeros.item()
        else:
            # print("INFO call _deterministic_z")
            expected_num_zeros = self.mask_size - self.target_mask_size 
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print("num of zeros is nan....")
            sys.exit()
        soft_mask = torch.sigmoid(z_loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(self.z_loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask
    
    def deterministic_z(self):
        # print("deterministic_z")
        # print(f"eval_target_model {self.eval_target_model}")
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga).reshape(*self.mask_output_shape)
        else:
            z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
            z = []
            for i in range(z_loga.shape[0]):
                z_ = self._deterministic_z(z_loga[i])
                z.append(z_)
            z = torch.stack(z).reshape(*self.mask_output_shape)
        return z
    
    def forward(self):
        # print("forward")
        # func = self.sample_z if (self.training and self.use_samlpe) else self.deterministic_z
        func = self.sample_z if self.training else self.deterministic_z
        z = func(self.z_loga).reshape(self.mask_output_shape)
        return z            
            
    def constrain_parameters(self):
        self.z_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def calculate_expected_score_sparsity(self):
        score = 1 - self.cdf_qz()
        sparsity = 1 - score.sum(-1) / self.mask_size
        return score, sparsity
 
# num_layers -> num_hidden_layers
# self.base_model_info.params_per_head self.base_model_info.dim_per_head # TODO
class L0Module(nn.Module):
    def __init__(self, cfg, device):
        super(L0Module, self).__init__()

        # base and target model info
        # n_matrix_mlp = 2 if "pythia" in cfg.name else 3
        n_matrix_mlp = 3
        l0_module_cfg=cfg
        self.base_model_info = self.set_model_info(l0_module_cfg.base_model, n_matrix_mlp=n_matrix_mlp) 
        # l0_module_cfg = cfg.l0_module
        self.target_model_info = None
        target_model_cfg = getattr(l0_module_cfg, "target_model", None)
        if target_model_cfg is not None:
            self.target_model_info = self.set_model_info(target_model_cfg, n_matrix_mlp=n_matrix_mlp)
        
        # l0 config
        self.pruning_modules = l0_module_cfg.pruning_modules        
        self.start_sparsity = l0_module_cfg.start_sparsity 
        # self.lagrangian_warmup_steps = Time.from_timestring(l0_module_cfg.lagrangian_warmup_steps).value # TODO
        self.lagrangian_warmup_steps = getattr(l0_module_cfg, "lagrangian_warmup_steps", 0)
        self.device = device
        # self.eval_target_model = l0_module_cfg.get("eval_target_model", True)
        self.eval_target_model = getattr(l0_module_cfg, "eval_target_model", True)
        
        # l0 params
        self.lambdas = {}
        self.lambdas["lambda_1"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.lambdas["lambda_2"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.masks = {}
        print(f"pruning modules: {self.pruning_modules}")
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        self.masks = torch.nn.ModuleDict(self.masks)
        self.lambdas = torch.nn.ParameterDict(self.lambdas)
        
        # config after initialization
        self.prunable_model_size = self.calculate_prunable_model_size(self.base_model_info)
        if target_model_cfg is not None:
            self.prunable_target_model_size = self.calculate_prunable_model_size(self.target_model_info)
            self.target_sparsity = 1 - self.prunable_target_model_size / self.prunable_model_size
        else:
            self.target_sparsity = l0_module_cfg.target_sparsity

        print("********** Initializing L0 Module **********") 
        for pruning_module in self.pruning_modules:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print(f"size", self.masks[pruning_module].mask_size)
        print(f"prunable model size: {self.prunable_model_size}")
        
    
    def set_model_info(self, cfg, n_matrix_mlp):
        ns = NS() 
        ns.hidden_size = cfg.d_model
        ns.intermediate_size = cfg.intermediate_size
        ns.num_attention_heads = cfg.n_heads
        ns.num_kv_heads = cfg.n_kv_heads
        ns.num_key_value_groups = cfg.n_heads // cfg.n_kv_heads
        ns.mlp_num_per_layer = 1
        ns.dim_per_head = ns.hidden_size // ns.num_attention_heads 
        ns.num_layers = cfg.n_layers
        ns.vocab_size = cfg.vocab_size

        # ns.params_per_head_layer = ns.hidden_size * ns.hidden_size * 4
        ns.params_per_head_layer = ns.hidden_size * ns.hidden_size * 2 + ns.hidden_size * ns.dim_per_head * ns.num_kv_heads * 2
        ns.params_per_head =  ns.params_per_head_layer // ns.num_attention_heads
        ns.params_per_mlp_layer = ns.hidden_size * ns.intermediate_size * n_matrix_mlp
        ns.params_per_intermediate_dim = ns.params_per_mlp_layer // ns.intermediate_size

        ns.full_model_size = (ns.params_per_head_layer + ns.params_per_mlp_layer) * ns.num_layers
        return ns
    
    def set_mask_eval_target_model(self, flag: bool):
        for key in self.masks:
            self.masks[key].eval_target_model = flag
        
    def set_mask_use_samlpe(self, flag: bool):
        for key in self.masks:
            self.masks[key].use_samlpe = flag

    def calculate_prunable_model_size(self, ns: NS):
        prunable_mlp_size = ns.params_per_mlp_layer * ns.num_layers
        prunable_head_layer_size = ns.params_per_head_layer * ns.num_layers
        prunable_model_size = 0
        if "hidden" in self.pruning_modules:
            return prunable_mlp_size + prunable_head_layer_size
        if "head_layer" in self.pruning_modules or "head" in self.pruning_modules:
            prunable_model_size += prunable_head_layer_size
        if "mlp" in self.pruning_modules or "intermediate" in self.pruning_modules:
            prunable_model_size += prunable_mlp_size
        return prunable_model_size
        
    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()
            
    def initialize_hidden(self):
        mask_shape = [self.base_model_info.hidden_size]
        num_params_per_mask=self.base_model_info.hidden_size * 4 + self.base_model_info.hidden_size * 4 * 2 # FIXME
        
        target_hidden_sparsity = None; pd=None; target_mask_size=None; 
        if self.target_model_info is not None:
            target_hidden_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            target_mask_size = self.target_model_info.hidden_size
            pd = {"lambda_1_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        hidden_mask = Mask(name="hidden",
                           mask_shape=mask_shape,
                           num_params_per_mask=num_params_per_mask,
                           mask_output_shape=[self.base_model_info.hidden_size],
                           target_sparsity=target_hidden_sparsity,
                           target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["hidden"] = hidden_mask

    def initialize_head(self):
        mask_shape = [self.base_model_info.num_layers * self.base_model_info.num_kv_heads, self.base_model_info.num_key_value_groups] # 剪枝后每个KV头对应的Q头数量应该相等
        num_params_per_mask = self.base_model_info.params_per_head
        mask_output_shape = [self.base_model_info.num_layers * self.base_model_info.num_kv_heads, 1, self.base_model_info.num_key_value_groups, 1] 
        
        target_head_sparsity = None; pd = {} ; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_sparsity = 1 - self.target_model_info.num_attention_heads / self.base_model_info.num_attention_heads
            target_mask_size = self.target_model_info.num_key_value_groups
            pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        head_mask = Mask(name="head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_head_sparsity,
                         target_mask_size=target_mask_size,
                            device=self.device,
                            eval_target_model=self.eval_target_model)
        self.masks["head"] = head_mask 

    # # original version
    # def initialize_head(self):
    #     mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads]
    #     num_params_per_mask = self.base_model_info.params_per_head
    #     mask_output_shape = [self.base_model_info.num_layers, 1, self.base_model_info.num_attention_heads, 1] 
        
    #     target_head_sparsity = None; pd = {} ; target_mask_size=None; 
    #     if self.target_model_info is not None:
    #         target_head_sparsity = 1 - self.target_model_info.num_attention_heads / self.base_model_info.num_attention_heads
    #         target_mask_size = self.target_model_info.num_attention_heads
    #         pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
    #               "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
    #         self.lambdas.update(pd)
    #     head_mask = Mask(name="head",
    #                      mask_shape=mask_shape,
    #                      num_params_per_mask=num_params_per_mask,
    #                      mask_output_shape=mask_output_shape,
    #                      target_sparsity=target_head_sparsity,
    #                      target_mask_size=target_mask_size,
    #                        device=self.device,
    #                        eval_target_model=self.eval_target_model)
    #     self.masks["head"] = head_mask 

    # def initialize_qk_head_dim(self): # only campatible when target model info is available
    #     # mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads, self.base_model_info.dim_per_head]
    #     mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_kv_heads, self.base_model_info.dim_per_head]
    #     # num_params_per_mask = 2 * self.base_model_info.hidden_size
    #     num_params_per_mask = (1 + self.base_model_info.num_key_value_groups) * self.base_model_info.hidden_size
    #     # mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
    #     mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.num_kv_heads * self.base_model_info.dim_per_head]
        
    #     target_qk_head_dim_sparsity = None; pd = {} 
    #     if self.target_model_info is not None:
    #         target_qk_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
    #         pd = {"lambda_1_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
    #               "lambda_2_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
    #         self.lambdas.update(pd)
    #     qk_head_dim = Mask(name="qk_head_dim",
    #                      mask_shape=mask_shape,
    #                      num_params_per_mask=num_params_per_mask,
    #                      mask_output_shape=mask_output_shape,
    #                      target_sparsity=target_qk_head_dim_sparsity,
    #                     #  target_mask_size=self.target_model_info.hidden_size,
    #                      target_mask_size=self.target_model_info.dim_per_head,
    #                      device=self.device)
    #     self.masks["qk_head_dim"] = qk_head_dim 
          
          
    # def initialize_vo_head_dim(self): # only campatible when target model info is available
    #     # mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads, self.base_model_info.dim_per_head]
    #     mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_kv_heads, self.base_model_info.dim_per_head]
    #     # num_params_per_mask = 2 * self.base_model_info.hidden_size
    #     num_params_per_mask = (1 + self.base_model_info.num_key_value_groups) * self.base_model_info.hidden_size
    #     # mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
    #     mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.num_kv_heads * self.base_model_info.dim_per_head]
        
    #     target_vo_head_dim_sparsity = None; pd = {} 
    #     if self.target_model_info is not None:
    #         target_vo_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
    #         pd = {"lambda_1_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
    #               "lambda_2_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
    #         self.lambdas.update(pd)
    #     vo_head_dim = Mask(name="vo_head_dim",
    #                      mask_shape=mask_shape,
    #                      num_params_per_mask=num_params_per_mask,
    #                      mask_output_shape=mask_output_shape,
    #                      target_sparsity=target_vo_head_dim_sparsity,
    #                      target_mask_size=self.target_model_info.dim_per_head,
    #                      device=self.device)
    #     self.masks["vo_head_dim"] = vo_head_dim 
        
    def initialize_head_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_head_layer
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_head_layer_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        head_layer_mask = Mask(name="head_layer",
                              mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                               target_sparsity=target_head_layer_sparsity,
                               target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["head_layer"] = head_layer_mask
        
    def initialize_intermediate(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.intermediate_size]
        num_params_per_mask=self.base_model_info.params_per_intermediate_dim
        mask_output_shape = [self.base_model_info.num_layers, 1, 1, self.base_model_info.intermediate_size] 
        
        target_int_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_int_sparsity = 1 - self.target_model_info.intermediate_size / self.base_model_info.intermediate_size
            target_mask_size = self.target_model_info.intermediate_size
            pd = {"lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["intermediate"] = int_mask
       

    def initialize_mlp(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_mlp_layer
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_mlp_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_mlp_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        mlp_mask = Mask(name="mlp",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_mlp_sparsity,
                        target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["mlp"] = mlp_mask 

    def initialize_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        # num_params_per_mask=self.base_model_info.params_per_head * self.base_model_info.num_attention_heads + self.base_model_info.params_per_mlp_layer
        num_params_per_mask=self.base_model_info.params_per_head_layer + self.base_model_info.params_per_mlp_layer
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_layer_sparsity = None; target_mask_size=None;  pd = {}
        if self.target_model_info is not None:
            target_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        layer_mask = Mask(name="layer",
                         mask_shape=mask_shape,
                          num_params_per_mask=num_params_per_mask,
                          mask_output_shape=mask_output_shape,
                          target_sparsity=target_layer_sparsity,
                          target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model) 
        self.masks["layer"] = layer_mask 
    
    def constrain_parameters(self):
        for key in self.masks:
            self.masks[key].constrain_parameters()

    def calculate_expected_score_sparsity(self):
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_expected_score_sparsity()
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys
    
    def transform_scores_for_head(self, expected_scores: dict):
        head_score = expected_scores["head"] # 12 * 12

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores["head_layer"]
        elif "layer" in expected_scores:
            head_layer_score = expected_scores["layer"] # 12
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1) # 12 * 1
       
        return head_layer_score, head_score

    def transform_scores_for_mlp(self, expected_scores: dict):
        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores["mlp"] # 12
        elif "layer" in expected_scores:
            mlp_score = expected_scores["layer"] # 12
        if mlp_score is not None:
            mlp_score = mlp_score.unsqueeze(-1)
        
        intermediate_score = expected_scores["intermediate"] # 12 * 3072
        return mlp_score, intermediate_score


    def get_expected_num_params(self, expected_scores: dict): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 
        # 12 * 12
        head_layer_score, head_score = self.transform_scores_for_head(expected_scores)
        # 将head_score由[self.base_model_info.num_layers * self.base_model_info.num_kv_heads, self.base_model_info.num_key_value_groups]变为[self.base_model_info.num_layers,self.base_model_info.num_kv_heads*self.base_model_info.num_key_value_groups]
        # 原始的head_score中，第0维的顺序为：layer0_kv_head0, layer0_kv_head1, layer0_kv_head2, layer0_kv_head3, layer1_kv_head0, layer1_kv_head1, layer1_kv_head2, layer1_kv_head3, ...
        head_score = head_score.view(self.base_model_info.num_layers, -1) # FIXME
        mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)

        head_score = (head_layer_score * head_score) # 12 * 12
        int_score = (mlp_score * int_score) # 12 * 3072

        # qk_score = None
        # if "qk_head_dim" in expected_scores:
        #     qk_head_dim_score = expected_scores["qk_head_dim"] # num_layer * hidden_size
        #     vo_head_dim_score = expected_scores["vo_head_dim"] # num_layer * hidden_size
        #     qk_head_dim_score = qk_head_dim_score.view(qk_head_dim_score.shape[0], -1) # 12 * 768
        #     vo_head_dim_score = vo_head_dim_score.view(vo_head_dim_score.shape[0], -1) # 12 * 768
        #     head_score = torch.repeat_interleave(head_score, self.base_model_info.dim_per_head, dim=1) # 12 * 768

        #     qk_score = head_score * qk_head_dim_score # 12 * 768
        #     vo_score = head_score * vo_head_dim_score # 12 * 768
                
        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"] # 768 
            
            # FIXME
            # if qk_score is None:
                # 在不处理head、qk_head_dim、vo_head_dim的情况下，head_score只会在这里被使用 
            num_parameters += torch.outer(hidden_score, head_score.reshape(-1)).sum() * self.masks.head.num_params_per_mask / self.base_model_info.hidden_size # 768 * 144
            num_parameters += torch.outer(hidden_score, int_score.reshape(-1)).sum() * self.masks.intermediate.num_params_per_mask / self.base_model_info.hidden_size # 768 * 36864
            # else:
            #     # num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * 2 # 12 * 768 * 768
            #     # num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * 2 # 12 * 768 * 768
            #     num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * (1+self.base_model_info.num_key_value_groups)
            #     num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * (1+self.base_model_info.num_key_value_groups)
            #     num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), int_score.unsqueeze(1))) * 3 # 12 * 768 * 3072
        else:
            num_parameters += torch.sum(head_score) * self.masks.head.num_params_per_mask
            num_parameters += torch.sum(int_score) * self.masks.intermediate.num_params_per_mask
        return num_parameters
    
    def get_target_sparsity(self, pruned_steps: int, full_sparsity: float = None):
        target_sparsity = full_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            target_sparsity = (target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup_steps) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps: int):
        def _lag_loss(expected_sparsity: torch.tensor, target_sparsity: float, lambda_1: torch.tensor, lambda_2: torch.tensor):
            lagrangian_loss = lambda_1 * (expected_sparsity - target_sparsity) + lambda_2 * (expected_sparsity - target_sparsity) ** 2 
            lagrangian_loss = lagrangian_loss.mean()
            return lagrangian_loss

        target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity)            
        expected_scores, expected_sparsitys = self.calculate_expected_score_sparsity()
        expected_size = self.get_expected_num_params(expected_scores) #! calculate \bar s
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        
        # return_v = {}
        if self.target_model_info is None:
            lagrangian_loss = _lag_loss(expected_sparsity, target_sparsity, self.lambdas["lambda_1"], self.lambdas["lambda_2"])
            return_v = {"expected_sparsity": expected_sparsity.item(), "target_sparsity": target_sparsity}
            for key in expected_sparsitys:
                return_v[f"expected_{key}_sparsity"] = expected_sparsitys[key].mean().item()
        else:
            lagrangian_loss = 0
            return_v = {}
            for pruning_module in self.pruning_modules:
                ts = self.get_target_sparsity(pruned_steps, self.masks[pruning_module].target_sparsity)
                expected_ts = expected_sparsitys[pruning_module] 
                lagrangian_loss += _lag_loss(expected_ts, ts, self.lambdas[f"lambda_1_{pruning_module}"], self.lambdas[f"lambda_2_{pruning_module}"])
                expected_ts = expected_ts.mean().item()
                return_v.update({"expected_{}_sparsity".format(pruning_module): expected_ts, "target_{}_sparsity".format(pruning_module): ts})
            return_v["expected_sparsity"] = expected_sparsity.item()
            return_v["target_sparsity"] = target_sparsity

        # 将return_v中元素转换为tensor
        for key in return_v:
            if torch.is_tensor(return_v[key]): continue
            return_v[key] = torch.tensor(return_v[key]).to(torch.bfloat16)
        return lagrangian_loss, return_v
    
        # return LagrangianOutput(loss=lagrangian_loss, **return_v)
        # ret = LagrangianOutput(loss=lagrangian_loss)
        # for key in ret:
        #     if isinstance(ret[key], torch.Tensor):
        #         ret[key] = ret[key].detach().requires_grad_()  # 确保它仍然需要梯度
        #         ret[key].retain_grad()  # 保留计算图，避免在第二次反向传播时报错
        # return ret
 
    # def forward(self, calculate_lagrangian=False, pruned_steps=0):
    #     def forward_fn(*inputs):
    #         return self.actual_forward(inputs[0], inputs[1])  # 这里调用真正的计算逻辑

    #     calculate_lagrangian_tensor = torch.tensor(calculate_lagrangian, dtype=torch.bfloat16, device=self.device, requires_grad=True)
    #     pruned_steps_tensor = torch.tensor(pruned_steps, dtype=torch.bfloat16, device=self.device, requires_grad=True)

    #     return checkpoint.checkpoint(forward_fn, calculate_lagrangian_tensor, pruned_steps_tensor)


    def forward(self, calculate_lagrangian: bool = False, pruned_steps: int = 0):
        self.constrain_parameters()
        if calculate_lagrangian:
            return self.lagrangian_regularization(pruned_steps)
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
        
        if "layer" in self.pruning_modules:
            zs.pop("layer_z")
            zs["mlp_z"] = []
            zs["head_layer_z"] = []

        if self.training:
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                z = mask.sample_z()
                # z = mask.deterministic_z() # FIXME
                zs[f"{pruning_module}_z"] = z
        else: # removed layerwise! 
            # print("self.training")
            with torch.no_grad():
                for pruning_module in self.pruning_modules:
                    mask = self.masks[pruning_module]
                    z = mask.deterministic_z()
                    zs[f"{pruning_module}_z"] = z
        if "layer_z" in zs:
            zs["mlp_z"] = zs.pop("layer_z")
            zs["head_layer_z"] = zs["mlp_z"]
        if "head_z" in zs:
            zs["head_z"]=zs["head_z"].view(self.base_model_info.num_layers, -1)

        # 将zs中所有浮点类型的tensor转换为bf16:
        for key in zs:
            if torch.is_tensor(zs[key]) and zs[key].dtype in [torch.float16, torch.float32]:
                zs[key] = zs[key].to(torch.bfloat16)

        # zs = zsOutput(**zs)
                
        # FIXME: 这部分代码可以避免gradient_checkpointing=True时的报错，但我不知道为什么。报错的内容如下：
        # RuntimeError: Trying to backward through the graph a second time 
        # (or directly access saved tensors after they have already been freed). 
        # Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). 
        # Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
        # for key in zs:
        #     if isinstance(zs[key], torch.Tensor):
        #         zs[key] = zs[key].detach().requires_grad_()  # 确保它仍然需要梯度
        #         zs[key].retain_grad()  # 保留计算图，避免在第二次反向传播时报错

        return zs


# def test_l0_module():
#     from omegaconf import OmegaConf as om 
#     cfg = om.load("/scratch/gpfs/mengzhou/space2/examples/examples/llm/yamls/llama/7b.yaml")
#     cfg = om.load("/scratch/gpfs/mengzhou/space2/examples/examples/llm/yamls/pythia/410m.yaml")
#     cfg.model.l0_module.pruning_modules = ["layer", "head", "intermediate", "hidden"]
    
#     l0_module = L0Module(cfg.model, "cpu")
    
#     # test run_through
#     print("\n***************************** \n run forward pass during training")
#     l0_module.train()
#     zs = l0_module.forward(calculate_lagrangian=False)
#     for key in zs:
#         print(key, zs[key].shape)

#     print("\n***************************** \n run forward pass during eval")
#     l0_module.eval()
#     zs = l0_module.forward(calculate_lagrangian=False)
#     for key in zs:
#         print(key, zs[key].shape)

#     print("\n***************************** \n run forward pass during lagrangian")
#     l0_module.train()
#     loss, v = l0_module(calculate_lagrangian=True, pruned_steps=320)
#     print("loss", loss.item())
#     for key in v:
#         if torch.is_tensor(v[key]): vv = v[key].item()
#         else: vv = v[key]
#         print(key, vv)
    
#     print("\n***************************** \n Test target sparsity") 
#     # test target_sparsity
#     target_sparsity = l0_module.get_target_sparsity(50, l0_module.target_sparsity)
#     print("target sparsity at step 50: ", target_sparsity)
     
#     target_sparsity = l0_module.get_target_sparsity(100, l0_module.target_sparsity)
#     print("target sparsity at step 100: ", target_sparsity)
    
#     target_sparsity = l0_module.get_target_sparsity(200, l0_module.target_sparsity)
#     print("target sparsity at step 200: ", target_sparsity)
#     import pdb; pdb.set_trace()


  
# if __name__ == "__main__":
#     test_l0_module()
    