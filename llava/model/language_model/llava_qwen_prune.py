# #    Copyright 2024 Hao Zhang
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.


# from typing import List, Optional, Tuple, Union, Dict
# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss

# import transformers
# from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.generation.utils import GenerateOutput

# from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
# from transformers import Qwen2Config
# from llava.model.modeling_qwen2_pruning import Qwen2ModelPrune, Qwen2ForCausalLMPrune

# from transformers.pytorch_utils import prune_linear_layer

# class LlavaQwenConfig(Qwen2Config):
#     model_type = "llava_qwen"


# class LlavaQwenModel(LlavaMetaModel, Qwen2ModelPrune):
#     config_class = LlavaQwenConfig

#     def __init__(self, config: Qwen2Config):
#         super(LlavaQwenModel, self).__init__(config)


# class LlavaQwenForCausalLM(Qwen2ForCausalLMPrune, LlavaMetaForCausalLM):
#     config_class = LlavaQwenConfig

#     def __init__(self, config):
#         super(Qwen2ForCausalLMPrune, self).__init__(config)
#         config.rope_scaling = None
#         self.model = LlavaQwenModel(config)
        
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#         print("[LlavaQwenForCausalLM] WARNING initialize_l0_module")
#         from dataclasses import dataclass, field

#         @dataclass
#         class TargetModelArguments:
#             d_model: int = 896
#             intermediate_size: int = 4864
#             n_heads: int = 14
#             n_kv_heads: int = 2
#             n_layers: int = 19
#             vocab_size: int = 151936
#         # class TargetModelArguments:
#         #     d_model: int = 640
#         #     intermediate_size: int = 3684
#         #     n_heads: int = 10
#         #     n_kv_heads: int = 2
#         #     n_layers: int = 20
#         #     vocab_size: int = 151936

#         @dataclass
#         class BaseModelArguments:
#             d_model: int = 896
#             intermediate_size: int = 4864
#             n_heads: int = 14
#             n_kv_heads: int = 2
#             n_layers: int = 22
#             vocab_size: int = 151936

#         from dataclasses import field
#         # from typing import List

#         @dataclass
#         class PruningArguments:
#             lagrangian_warmup_steps: int = 500
#             pruning_modules: List[str] = field(default_factory=lambda: ["intermediate", "layer", "hidden", "head"])
#             start_sparsity: float = 0.0 # TODO
#             target_sparsity: float = 0.5 # TODO
#             eval_target_model: bool = False
#             init_device: str = 'cpu'
#             base_model: BaseModelArguments = None
#             target_model: TargetModelArguments = None

#         base_model_cfg = BaseModelArguments()
#         target_model_cfg = TargetModelArguments()
#         l0_module_cfg = PruningArguments(base_model=base_model_cfg, target_model=target_model_cfg)
#         l0_module_cfg.target_model = target_model_cfg
#         l0_module_cfg.base_model = base_model_cfg
#         self.get_model().initialize_l0_module(l0_module_cfg)

#     def get_model(self):
#         return self.model

#     # def prune_params(self, zs=None):
#     #     if zs is None:
#     #         self.l0_module.eval()
#     #         zs = self.l0_module(calculate_lagrangian=False)
#     #     # wte as well :) 
#     #     # ln_f if hidden states are to be pruned
#     #     if "hidden_z" in zs:
#     #         hidden_z = zs["hidden_z"]
#     #         remaining_index = torch.where(~hidden_z.eq(0))[0]
#     #         self.transformer.ln_f.prune_params(hidden_z)
#     #         self.transformer.wte.weight.data = self.transformer.wte.weight.data.mul(hidden_z)
#     #         self.transformer.wte.weight = torch.nn.parameter.Parameter(
#     #             self.transformer.wte.weight.index_select(1, remaining_index).clone())
#     #         self.transformer.wte.embedding_dim = len(remaining_index)
#     #         # This is def a bug in llama, but does not incur too much issue
#     #         self.transformer.output.weight.data = self.transformer.output.weight.data.mul(hidden_z) 
#     #         half = self.transformer.output.weight.data.dtype == torch.float16
#     #         self.transformer.output = prune_linear_layer(self.transformer.output, remaining_index, dim=1)
#     #         if half:
#     #             self.transformer.output = self.transformer.output.half()
            
#     #     for i, block in enumerate(self.transformer.blocks):
#     #         zs_block = self.get_zs_block(zs, i)
#     #         block.prune_params(zs_block)
        

#     def prune_params(self):
#         # self.model.prune_params() 
#         self.model.l0_module.eval()
#         # zs = self.model.l0_module(calculate_lagrangian=False)
#         hidden_z, remaining_index = self.get_model().prune_params()
        
#         # prune the lm_head
#         if hidden_z is not None:
#             # FIXME: Maybe we shouldn't multiply the lm_head with hidden_z
#             # self.lm_head.weight.data =  self.lm_head.weight.data.mul(hidden_z) 
#             self.lm_head = prune_linear_layer(self.lm_head, remaining_index, dim=1)

#             # prune the output dimension of the last layer of mmprojector
#             # FIXME: Maybe we shouldn't multiply the weight and bias with hidden_z
#             # self.get_model().mm_projector[-1].weight.data = self.get_model().mm_projector[-1].weight.data.transpose(0, 1).mul(hidden_z.squeeze(0)).transpose(0, 1)
#             # self.get_model().mm_projector[-1].bias.data = self.get_model().mm_projector[-1].bias.data.mul(hidden_z.squeeze(0))
#             self.get_model().mm_projector[-1]=prune_linear_layer(self.get_model().mm_projector[-1], remaining_index)
            
#             # prune the image_newline (special token embedding)
#             # FIXME: Maybe we shouldn't multiply the image_newline with hidden_z
#             # self.get_model().image_newline.data = self.get_model().image_newline.data.mul(hidden_z) # image_newline是Parameter，size为[hidden_size]
#             self.get_model().image_newline = torch.nn.parameter.Parameter(
#                 self.get_model().image_newline.index_select(0, remaining_index).clone())
        

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         image_sizes: Optional[List[List[int]]] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         pruned_steps: int = 0,
#         # pruned_steps: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         # print("LlavaQwenForCausalLM called", flush=True)
#         # print(f"pruned_steps LlavaQwenForCausalLM: {pruned_steps}", flush=True)
#         if inputs_embeds is None:
#             (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)

#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             pruned_steps=pruned_steps,
#         )

#     @torch.no_grad()
#     def generate(
#         self,
#         inputs: Optional[torch.Tensor] = None,
#         images: Optional[torch.Tensor] = None,
#         image_sizes: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#         position_ids = kwargs.pop("position_ids", None)
#         attention_mask = kwargs.pop("attention_mask", None)
#         if "inputs_embeds" in kwargs:
#             raise NotImplementedError("`inputs_embeds` is not supported")

#         if images is not None:
#             (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
#         else:
#             inputs_embeds = self.get_model().embed_tokens(inputs)

#         return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
#         images = kwargs.pop("images", None)
#         image_sizes = kwargs.pop("image_sizes", None)
#         inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
#         if images is not None:
#             inputs["images"] = images
#         if image_sizes is not None:
#             inputs["image_sizes"] = image_sizes
#         return inputs


# AutoConfig.register("llava_qwen", LlavaQwenConfig)
# AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
