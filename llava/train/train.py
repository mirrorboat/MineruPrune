# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import io
import copy
from dataclasses import dataclass, field
import json
import math
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

from docutils.utils.math.latex2mathml import mo
import torch
import numpy as np

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, process_anyres_image
# from llava.pruning_callback import PruningCallback
import random
import yaml
from petrel_client.client import Client
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

# @dataclass
# class TargetModelArguments:
#     d_model: int = 784 # = 14 * 56
#     intermediate_size: int = 4096
#     n_heads: int = 14
#     n_kv_heads: int = 2
#     n_layers: int = 23
#     vocab_size: int = 151936

# @dataclass
# class TargetModelArguments:
#     d_model: int = 672 # = 14 * 48
#     intermediate_size: int = 3684
#     n_heads: int = 14
#     n_kv_heads: int = 2
#     n_layers: int = 20
#     vocab_size: int = 151936

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
    n_layers: int = 19
    vocab_size: int = 151936

@dataclass
class BaseModelArguments:
    d_model: int = 896 # 14 * 64
    intermediate_size: int = 4864
    n_heads: int = 14
    n_kv_heads: int = 2
    n_layers: int = 22
    vocab_size: int = 151936
# 896/14=64
from dataclasses import field
# from typing import List

@dataclass
class PruningArguments:
    # lagrangian_warmup_steps: int = 1024
    lagrangian_warmup_steps: int = 500
    # pruning_modules: List[str] = field(default_factory=lambda: ["intermediate", "layer", "hidden", "head"])
    pruning_modules: List[str] = field(default_factory=lambda: ["intermediate", "layer", "hidden", "head"])
    # pruning_modules: List[str] = field(default_factory=lambda: ["intermediate", "layer", "hidden", "qk_head_dim", "vo_head_dim"])
    start_sparsity: float = 0.0 # TODO
    target_sparsity: float = 0.5 # TODO
    eval_target_model: bool = False
    init_device: str = 'cpu'
    base_model: BaseModelArguments = None
    target_model: TargetModelArguments = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_box_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None,
                                metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    mm_modify: bool = True

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    save_vision_tower: bool = field(default=False)
    tune_entire_model: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_projector: bool = field(default=False)
    tune_l0_module_only: bool = field(default=False)
    max_grad_norm: Optional[float] = field(default=1.0)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_vision_tower_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lag_lr: Optional[float] = 1.0
    use_hook: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if getattr(trainer.args, 'save_vision_tower', False):
        if trainer.deepspeed:
            torch.cuda.synchronize()
        trainer.model.get_vision_tower().image_processor.save_pretrained(
            os.path.join(output_dir, 'vision_tower'))
        trainer.model.get_vision_tower().config.save_pretrained(
                os.path.join(output_dir, 'vision_tower'))
        weight_to_save = get_vision_tower_state_maybe_zero_3(
            trainer.model.get_vision_tower().vision_tower.named_parameters())
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            torch.save(weight_to_save, os.path.join(
                output_dir, 'vision_tower/pytorch_model.bin'))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def proxy_off():
        os.environ['http_proxy'] = ''
        os.environ['HTTP_PROXY'] = ''
        
def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            if data_args.mm_modify:
                sentence["value"] = sentence["value"].replace("Grouding with reading order", "Grounding with reading order")
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token).strip()

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN_2
    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_opt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT
    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n'
        conversation = source[0]['value'] + source[1]['value'] + tokenizer.eos_token
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "opt":
        return preprocess_opt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen_2":
        return preprocess_qwen_2(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""


    def read_image(self, image_path):
        if "s3" in image_path:
            if "mineru:" not in image_path:
                image_path = "mineru:" + image_path
            image_bytes = self.client.get(image_path)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        return image

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = []
        proxy_off()
        self.client = Client()
        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "last" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".json"):
            with open(data_path, "r") as json_file:
                list_data_dict = json.load(json_file)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 100
        while attempt < max_attempt:
            try:
                data_dict = self._sample_item(i)
                break
            except Exception as e:
                print(f"Error loading data {i}, attempt {attempt}, data {self.list_data_dict[i]}")
                print(f"Exception: {str(e)}")
                
                attempt += 1
                i = random.randint(0, len(self.list_data_dict)-1)

        return data_dict

    def _sample_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            
            image = self.read_image(image_file)
            
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
                image_size = image.size
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints, self.data_args.image_aspect_ratio)
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, 1024, 1024)
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_sizes = [instance['image_size'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    eval_dataset = None
    if data_args.eval_data_path is not None:
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                             data_path=data_args.eval_data_path,
                                             data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PruningArguments, TargetModelArguments))
    model_args, data_args, training_args, l0_module_cfg, target_model_cfg = parser.parse_args_into_dataclasses()
    # from dataclasses import asdict
    # l0_module_cfg = asdict(l0_module_cfg)
    # target_model_cfg = asdict(target_model_cfg)
    base_model_cfg = BaseModelArguments()
    l0_module_cfg.target_model = target_model_cfg
    l0_module_cfg.base_model = base_model_cfg
    # l0_module_cfg.__dict__["target_model"] = target_model_cfg
    # model_args.__dict__["l0_module"] = l0_module_cfg
    # model_args.__dict__.update(vars(model_args))
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif 'opt' in model_args.model_name_or_path.lower():
            model = LlavaOPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                **bnb_model_from_pretrained_args
            )
        elif 'qwen' in model_args.model_name_or_path.lower():
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if 'qwen' in model_args.model_name_or_path.lower():
            tokenizer.unk_token = tokenizer.pad_token = "<|endoftext|>"
            tokenizer.pad_token_id = tokenizer.unk_token_id = 151643
        elif 'opt' in model_args.model_name_or_path.lower():
            tokenizer.unk_token = tokenizer.pad_token
            tokenizer.pad_token_id = tokenizer.unk_token_id = 1
        else:
            tokenizer.pad_token = tokenizer.unk_token
        
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # model.get_model().set_l0_module_eval(True) # FIXME
    # model_args.__dict__.update(vars(model_args))
    # l0_module_cfg.__dict__.update(vars(model.config))
    # for key, value in vars(model_args).items():
        # setattr(l0_module_cfg, key, value)
            
    # from transformers import AutoConfig
    # tmpconfig = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # 将tmpconfig的键值对更新到l0_module_cfg中
    # for key, value in vars(tmpconfig).items():
    #     l0_module_cfg[key] = value
        # setattr(l0_module_cfg, key, value)

    # breakpoint()

    # model.get_model().initialize_l0_module(l0_module_cfg)
            
    # model.get_model().set_l0_module_eval(True)
    # model.get_model().l0_module.set_mask_use_samlpe(False)
    # print("WARNING: L0 module is set to eval mode and deterministic mask!")

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        for p in model.get_model().l0_module.parameters():
            p.requires_grad = True

        if training_args.tune_l0_module_only:
            # model.requires_grad_(False)
            for p in model.get_model().l0_module.parameters():
                p.requires_grad = True
            model.requires_grad_(True)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            for p in vision_tower.parameters():
                p.requires_grad = False
            print("WARNING: Only tune the L0 module!")
        else:
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            if model_args.tune_mm_mlp_adapter:
                model.requires_grad_(False)
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False
                
            model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
            if training_args.unfreeze_mm_vision_tower:
                lr_of_vit = training_args.mm_vision_tower_lr if training_args.mm_vision_tower_lr is not None else training_args.learning_rate
                lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
                training_args.mm_projector_lr = lr_of_mlp
                training_args.mm_vision_tower_lr = lr_of_vit
                for p in vision_tower.parameters():
                    p.requires_grad = True
                rank0_print(
                    f'Tune the entire model! The LR of ViT is {training_args.mm_vision_tower_lr}. The LR of MLP is {training_args.mm_projector_lr}. The LR of LLM is {training_args.learning_rate}')

            model.config.tune_entire_model = training_args.tune_entire_model
            if training_args.tune_entire_model:
                model.requires_grad_(True)
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
                for p in vision_tower.parameters():
                    p.requires_grad = True

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_box_start_end = data_args.mm_use_box_start_end = model_args.mm_use_box_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_box_start_end = model_args.mm_use_box_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    data_args.max_input_len = training_args.model_max_length - vision_tower.num_patches
    print(f"Max input length: {data_args.max_input_len}")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    if training_args.use_hook:
        feature_ls2=[]

        def feature_hook2(module, input, output):
            if not model.training:
                # output_copy=output.clone().detach()
                output_copy=output[0].clone().detach()
                # output_copy=input[0].clone().detach()
                feature_ls2.append(output_copy)

        # model.get_model().layers[0].input_layernorm.register_forward_hook(feature_hook)
        # model.get_model().layers[0].self_attn.register_forward_hook(feature_hook2)
        # model.get_model().layers[0].register_forward_hook(feature_hook2)
        model.get_model().layers[-1].register_forward_hook(feature_hook2)


    for n, p in model.named_parameters():
        print(n, flush=True)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if training_args.use_hook:

        model.get_model().l0_module.set_mask_eval_target_model(True)
        # hidden_z = model.get_model().l0_module(calculate_lagrangian=False)["hidden_z"]
        # non_zero_idx = torch.where(~hidden_z.eq(0))[0]
        # print(torch.where(hidden_z.eq(0))[0])
        # print(f"length of non-zero idx: {len(non_zero_idx)}")
        # feature_ls2[0]形状为[1, token len, old dim]，根据non_zero_idx仅保留非零idx对应的dim

        # non_zero_idx = []
        # For head_hidden-only model
        # non_zero_idx = [  0,   1,   4,   6,   7,   9,  10,  11,  12,  13,  14,  15,  16,  18,
        #  19,  20,  21,  23,  24,  25,  27,  30,  31,  32,  34,  36,  38,  40,
        #  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  52,  56,  57,  59,
        #  60,  61,  62,  64,  65,  66,  67,  68,  69,  70,  72,  73,  74,  75,
        #  76,  77,  78,  79,  81,  82,  83,  85,  87,  88,  89,  90,  93,  95,
        #  96,  97,  98,  99, 100, 102, 103, 104, 105, 106, 107, 109, 110, 111,
        # 116, 117, 119, 120, 121, 123, 124, 126, 127, 128, 129, 132, 133, 134,
        # 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 149, 150, 151,
        # 152, 153, 154, 155, 156, 158, 159, 162, 163, 165, 166, 168, 169, 170,
        # 171, 173, 174, 175, 176, 177, 179, 181, 184, 185, 186, 187, 192, 193,
        # 194, 195, 196, 197, 199, 200, 202, 203, 204, 205, 206, 207, 208, 209,
        # 210, 213, 215, 216, 217, 218, 219, 221, 223, 224, 225, 226, 227, 228,
        # 229, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
        # 245, 246, 247, 248, 249, 252, 253, 254, 255, 257, 260, 261, 262, 264,
        # 265, 266, 267, 268, 271, 275, 276, 277, 278, 280, 283, 284, 285, 286,
        # 287, 288, 291, 292, 293, 294, 296, 297, 298, 299, 300, 301, 302, 303,
        # 304, 305, 306, 307, 308, 310, 313, 314, 315, 318, 319, 320, 322, 323,
        # 325, 326, 328, 329, 330, 331, 332, 334, 335, 337, 338, 340, 341, 342,
        # 346, 347, 348, 349, 351, 352, 353, 355, 356, 357, 358, 360, 361, 364,
        # 366, 367, 368, 370, 371, 373, 374, 375, 377, 380, 381, 383, 384, 385,
        # 386, 387, 388, 389, 390, 392, 393, 394, 395, 396, 397, 398, 399, 400,
        # 402, 403, 404, 407, 408, 410, 411, 412, 414, 415, 418, 420, 422, 423,
        # 424, 426, 427, 429, 430, 431, 432, 433, 434, 435, 437, 438, 439, 440,
        # 441, 442, 444, 445, 446, 447, 449, 450, 452, 453, 454, 456, 458, 459,
        # 462, 463, 465, 467, 468, 469, 470, 471, 472, 473, 474, 476, 478, 479,
        # 480, 482, 483, 484, 485, 486, 488, 489, 490, 492, 494, 496, 498, 500,
        # 501, 502, 505, 508, 509, 510, 511, 513, 515, 516, 517, 519, 522, 523,
        # 524, 525, 527, 528, 529, 533, 535, 538, 539, 540, 541, 543, 544, 546,
        # 547, 548, 549, 550, 551, 553, 554, 555, 557, 558, 559, 560, 561, 563,
        # 567, 568, 569, 570, 572, 573, 574, 575, 577, 578, 579, 580, 581, 582,
        # 584, 585, 586, 587, 590, 591, 592, 594, 596, 598, 599, 600, 601, 602,
        # 605, 607, 608, 609, 611, 612, 613, 614, 616, 617, 618, 619, 620, 624,
        # 626, 627, 628, 631, 634, 636, 638, 639, 641, 642, 643, 644, 646, 647,
        # 648, 649, 650, 652, 653, 655, 657, 658, 659, 660, 661, 663, 664, 665,
        # 666, 668, 669, 673, 674, 675, 676, 677, 680, 681, 682, 684, 685, 686,
        # 687, 690, 691, 692, 693, 695, 696, 697, 699, 700, 702, 703, 704, 705,
        # 706, 707, 708, 709, 710, 711, 713, 715, 717, 720, 722, 724, 727, 728,
        # 729, 730, 731, 732, 733, 734, 735, 736, 738, 740, 741, 742, 743, 744,
        # 745, 746, 747, 748, 751, 752, 753, 754, 756, 757, 758, 759, 760, 761,
        # 764, 766, 767, 768, 769, 771, 772, 773, 774, 775, 776, 777, 778, 780,
        # 781, 782, 783, 784, 785, 786, 787, 789, 790, 793, 794, 799, 800, 801,
        # 802, 803, 804, 806, 807, 809, 810, 811, 812, 813, 814, 816, 818, 819,
        # 820, 821, 824, 825, 826, 827, 829, 831, 832, 834, 836, 837, 838, 839,
        # 840, 841, 844, 845, 848, 849, 850, 851, 852, 854, 856, 857, 858, 859,
        # 861, 862, 863, 864, 868, 869, 870, 871, 872, 874, 876, 877, 879, 881,
        # 882, 883, 884, 885, 888, 889, 890, 892, 894, 895]

        # For 0409 model
        non_zero_idx = [  0,   1,   2,   6,   7,   8,   9,  11,  12,  13,  14,  15,  16,  18,
         19,  20,  21,  22,  23,  24,  25,  27,  28,  30,  31,  33,  34,  35,
         38,  40,  41,  42,  43,  45,  46,  47,  48,  49,  51,  56,  58,  60,
         61,  64,  65,  67,  68,  69,  70,  73,  74,  75,  76,  77,  78,  79,
         83,  84,  86,  87,  88,  89,  91,  93,  95,  96,  99, 100, 102, 103,
        104, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 124, 129, 131, 133, 134, 135, 136, 137, 138, 139, 140, 142,
        144, 145, 147, 148, 150, 152, 154, 155, 156, 158, 161, 162, 163, 165,
        166, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
        184, 185, 186, 188, 190, 191, 192, 193, 194, 195, 197, 198, 201, 202,
        203, 204, 205, 206, 207, 208, 209, 210, 212, 213, 215, 219, 220, 221,
        222, 224, 226, 227, 228, 231, 232, 233, 234, 235, 236, 237, 238, 240,
        241, 244, 245, 246, 248, 249, 251, 252, 253, 256, 257, 260, 261, 262,
        264, 265, 266, 267, 269, 271, 272, 275, 276, 278, 279, 280, 282, 283,
        284, 285, 286, 287, 288, 290, 291, 292, 293, 294, 296, 297, 298, 299,
        300, 301, 302, 303, 304, 305, 306, 307, 310, 311, 313, 315, 316, 318,
        319, 320, 322, 323, 324, 325, 326, 327, 328, 332, 333, 334, 335, 340,
        341, 342, 343, 345, 346, 347, 348, 350, 351, 352, 353, 357, 358, 359,
        360, 362, 363, 364, 366, 367, 368, 370, 371, 373, 374, 375, 376, 377,
        380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 391, 392, 393, 394,
        395, 397, 399, 400, 401, 402, 403, 404, 408, 410, 412, 414, 415, 419,
        420, 421, 422, 423, 424, 425, 426, 429, 430, 431, 432, 433, 434, 436,
        437, 438, 440, 441, 442, 445, 446, 447, 449, 451, 452, 453, 454, 456,
        457, 459, 460, 462, 463, 464, 465, 466, 467, 469, 470, 471, 472, 473,
        474, 476, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 490, 491,
        493, 494, 496, 497, 498, 499, 503, 504, 505, 507, 508, 510, 511, 513,
        514, 515, 516, 522, 524, 525, 526, 527, 528, 530, 531, 533, 534, 537,
        538, 539, 540, 541, 543, 544, 545, 546, 547, 548, 549, 551, 552, 553,
        554, 555, 556, 559, 560, 561, 562, 563, 565, 566, 567, 568, 569, 570,
        571, 572, 575, 577, 578, 580, 581, 582, 583, 584, 585, 586, 587, 590,
        591, 592, 593, 594, 595, 596, 599, 600, 601, 602, 603, 604, 605, 606,
        608, 609, 610, 611, 613, 614, 615, 617, 619, 620, 621, 624, 626, 627,
        629, 630, 631, 632, 633, 634, 636, 639, 641, 642, 643, 644, 646, 648,
        649, 650, 653, 654, 655, 656, 657, 659, 661, 663, 664, 665, 666, 667,
        668, 669, 671, 672, 674, 676, 677, 678, 680, 681, 682, 683, 684, 685,
        686, 690, 691, 692, 693, 694, 695, 696, 698, 699, 700, 703, 705, 706,
        707, 708, 709, 710, 711, 713, 714, 715, 716, 717, 719, 720, 721, 722,
        723, 724, 725, 726, 727, 729, 731, 732, 735, 736, 737, 738, 739, 740,
        741, 742, 743, 744, 745, 747, 748, 749, 751, 752, 753, 754, 757, 758,
        760, 761, 762, 764, 765, 766, 767, 768, 769, 770, 773, 774, 777, 778,
        779, 781, 784, 785, 786, 787, 788, 790, 793, 794, 795, 796, 797, 798,
        799, 800, 801, 802, 804, 805, 806, 808, 809, 810, 811, 813, 814, 816,
        817, 818, 820, 821, 822, 824, 826, 827, 829, 833, 834, 836, 837, 838,
        842, 845, 847, 848, 850, 851, 852, 853, 854, 855, 857, 858, 859, 861,
        862, 863, 864, 865, 866, 869, 870, 871, 872, 873, 874, 875, 876, 877,
        878, 880, 881, 882, 884, 887, 888, 891, 894, 895]
        feature_ls2[0]=feature_ls2[0][:, :, non_zero_idx]

        feature_ls2[0]=feature_ls2[0].to(torch.float32).cpu().numpy()
        np.save("layer-1_output.npy", feature_ls2[0])
        print(f"length of feature_ls2: {len(feature_ls2)}")
        exit()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
