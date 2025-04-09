import os
from .clip_encoder import CLIPVisionTower
from .sam_encoder import SAMVisionTower
from .sam_encoder_hd import SAMVisionTower_HD
from .sam_encoder_hhd import SAMVisionTower_HHD
from .siglip_encoder import SigLipVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    print(f"Building vision tower: {vision_tower.lower()}")
    if 'siglip' in vision_tower.lower():
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif 'sam-hhd' in vision_tower.lower():
        return SAMVisionTower_HHD(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'sam-hd' in vision_tower.lower():
        return SAMVisionTower_HD(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'sam' in vision_tower.lower():
        return SAMVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'clip' in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
