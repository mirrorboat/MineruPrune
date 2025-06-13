from llava.model import LlavaQwenForCausalLM
from dataclasses import dataclass, field
from typing import List
import torch
import collections

def change_keys(weights, output_file=None):
    """ rename the keys in the weight file to match the new model """
    exitsing_layers = []
    for key in weights:
        if "layers" in key and "vision_tower" not in key:
            layer = int(key[key.index("layers") + len("layers."):].split(".")[0])
            if layer not in exitsing_layers:
                exitsing_layers.append(layer)
    exitsing_layers = sorted(exitsing_layers)
    print("Existing layers: ", len(exitsing_layers), exitsing_layers)
    
    # new_weights = {}
    # new_weights改为collections.OrderedDict：
    new_weights = collections.OrderedDict()
    for key in weights:
        if "layers" in key and "vision_tower" not in key:
            layer_index = key.index("layers") + len("layers.")
            text_before_layer_index = key[:layer_index]
            layer = int(key[layer_index:].split(".")[0])
            text_after_layer_index = key[layer_index + len(str(layer)) + 1:]
            current_layer = exitsing_layers.index(layer)
            new_key = text_before_layer_index + str(current_layer) + "." + text_after_layer_index
            if key != new_key:
                print(f"{key} -> {new_key}")
                # print("Old param key:", key)
                # print("New param key:", new_key)
        else:
            new_key = key
        new_weights[new_key] = weights[key]
    if output_file is not None:
        torch.save(new_weights, output_file)
    else:
        return new_weights

# @dataclass
# class TargetModelArguments:
#     d_model: int = 640 # = 10 * 64
#     intermediate_size: int = 3684
#     n_heads: int = 10
#     n_kv_heads: int = 2
#     n_layers: int = 20
#     vocab_size: int = 151936
    
@dataclass
class TargetModelArguments:
    d_model: int = 896 # = 10 * 64
    intermediate_size: int = 4864
    n_heads: int = 14
    n_kv_heads: int = 2
    n_layers: int = 22
    vocab_size: int = 151936

@dataclass
class BaseModelArguments:
    d_model: int = 896 # 14 * 64
    intermediate_size: int = 4864
    n_heads: int = 14
    n_kv_heads: int = 2
    n_layers: int = 24
    vocab_size: int = 151936

# from typing import List

@dataclass
class PruningArguments:
    lagrangian_warmup_steps: int = 640
    pruning_modules: List[str] = field(default_factory=lambda: ["intermediate", "layer", "hidden", "head"])
    start_sparsity: float = 0.0 # TODO
    target_sparsity: float = 0.5 # TODO
    lagrangian_warmup_steps: int = 0 # TODO
    eval_target_model: bool = False # important
    # eval_target_model: bool = False
    init_device: str = 'cpu'
    base_model: BaseModelArguments = None
    target_model: TargetModelArguments = None

base_model_cfg = BaseModelArguments()
target_model_cfg = TargetModelArguments()
l0_module_cfg = PruningArguments(base_model=base_model_cfg, target_model=target_model_cfg)
l0_module_cfg.target_model = target_model_cfg
l0_module_cfg.base_model = base_model_cfg
# path="/mnt/petrelfs/chenjingzhou/cjz/MineruPrune/playground/training/training_dirs/head-checkpoint-1200"
path="/mnt/petrelfs/chenjingzhou/cjz/MineruPrune/playground/training/training_dirs/Qwen2-0.5B-Siglip-Pretrain-Qwen2-0.5B-Stage2-0427layer22_smooth/checkpoint-3200"
model = LlavaQwenForCausalLM.from_pretrained(path, attn_implementation="flash_attention_2")
# model.get_model().set_l0_module_eval(True)
# zs = model.get_model().l0_module(calculate_lagrangian=False)
del model.get_model().layers[22]
del model.get_model().layers[19]
# breakpoint()
# model.get_model().l0_module.initialize_qk_head_dim()
# model.get_model().l0_module.initialize_vo_head_dim()
# model.get_model().l0_module.initialize_intermediate()
# model.get_model().l0_module.initialize_hidden()
# model.get_model().l0_module.initialize_layer()
# model.get_model().initialize_l0_module(l0_module_cfg)
# model.load_state_dict(torch.load(path+"/pytorch_model.bin"))
# breakpoint()
# model.get_model().config[""]
# 打印模型各部分的dtype:
# for name, param in model.named_parameters():
#     print(name, param.dtype)
# model.get_model().prune_params() 

# model.prune_params() 
# breakpoint()
model.get_model().l0_module = None
# breakpoint()
model_state_dict = model.state_dict()
# breakpoint()
model_state_dict=change_keys(model_state_dict)

# 直接保存model_state_dict为bin文件
torch.save(model_state_dict, "pytorch_model0428.bin")
