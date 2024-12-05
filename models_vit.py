# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
            
            
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with backbone freezing support."""
    def __init__(self, global_pool=False, freeze_backbone=False, num_classes=1000, **kwargs):
        super().__init__(**kwargs)
        
        self.global_pool = global_pool
        self.freeze_backbone = freeze_backbone
        
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        # Define a classification head (MLP)
        self.head = nn.Linear(kwargs['embed_dim'], num_classes)

        # Freeze the backbone if required
        if self.freeze_backbone:
            self._freeze_backbone()

    def forward_features(self, x):
        """Extract features from the backbone."""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand class token
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # Global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        """Forward pass through the model."""
        x = self.forward_features(x)
        x = self.head(x)  
        return x

    def _freeze_backbone(self):
        """Freeze the foundational model (backbone)."""
        for name, param in self.named_parameters():
            if "head" not in name and "fc_norm" not in name:
                param.requires_grad = False
    
    
    def train(self, mode=True):
        """
        Override the `train` method to set the backbone to eval mode
        and the head to training mode.
        """
        super().train(mode)  # Set the entire model to training mode
        
        # Set backbone (foundational model) to eval mode
        if self.freeze_backbone:
            for module_name, module in self.named_children():
                if module_name not in ['head', 'fc_norm']:
                    module.eval()
        
        # Ensure head is always in training mode when `mode=True`
        if mode:
            self.head.train()
            # self.fc_norm.train()


def vit_large_patch16(freeze_backbone, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, freeze_backbone=freeze_backbone,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


