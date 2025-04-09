# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

import transformers
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import SamVisionConfig
from transformers import SamProcessor, SamImageProcessor
from transformers import SamPreTrainedModel, SamModel
from transformers.models.sam.modeling_sam import SamVisionLayer, SamVisionEncoder
from PIL import Image

class SamVisionModel(SamPreTrainedModel):
    config_class = SamVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SamVisionLayer"]

    def __init__(self, config: SamVisionConfig):
        super().__init__(config)
        self.vision_encoder = SamVisionEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class SAMVisionTower_HHD(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.args = args
        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.input_image_size = getattr(args, 'input_image_size', 1024)
        self.pixel_shuffle = getattr(args, 'add_pixel_shuffle', False)
        
        if not delay_load:
            print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        else:
            self.cfg_only = SamVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            return

        self.image_processor = SamProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.preprocess = self.image_processor.__call__
        self.image_processor.image_mean = [0.485,0.456,0.406]

        self.vision_tower = SamVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device)).last_hidden_state.to(device=self.device)

        if self.pixel_shuffle:
            image_features = nn.functional.pixel_unshuffle(image_features, 2)
        
        image_features = image_features.flatten(2, 3).transpose(1, 2)
        
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # hard code
        if self.pixel_shuffle:
            return 256 * 4
        else:
            return 256

    @property
    def num_patches(self):
        # hard code
        if self.pixel_shuffle:
            return 32 * 32
        else:
            return 64 * 64