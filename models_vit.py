# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from torchvision import models
            
            
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
        print(f"\033[94m\tFoundation model freezed\033[0m")
    
    
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


class FusionVisionTransformer(VisionTransformer):
    def __init__(self, global_pool=False, freeze_backbone=False, no_visuals=False, metadata_len=None, num_classes=1000, **kwargs):
        super().__init__(global_pool, freeze_backbone, num_classes, **kwargs)
        
        # self.head = nn.Linear(kwargs['embed_dim']*2, num_classes)
        self.metadata_len = metadata_len
        self.no_visuals = no_visuals
        
        self.fc_projector = nn.Linear(self.metadata_len, kwargs['embed_dim'])
        size_mult = 2-int(self.no_visuals) 
        self.head =  nn.Sequential(
                    nn.Linear(kwargs['embed_dim']*size_mult, 512),  
                    nn.ReLU(),                        
                    nn.Linear(512, 128), 
                    nn.ReLU(),                      
                    nn.Linear(128, num_classes)  
                )
        
        
    
    def forward(self, x, m):
        """Forward pass through the model."""
        m = self.fc_projector(m)  
        if not self.no_visuals:
            x = self.forward_features(x)
            return self.head(torch.cat((x, m), 1))
        
        return self.head(m)


# ResNet Model
def resnet(use_metadata, freeze_backbone, metadata_len, **kwargs):
    assert not use_metadata, "ResNet baseline does not use metadata"
    
    model = models.resnet50(pretrained=True)
    
    # Modify the final fully connected layer to output 2 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Optionally freeze the backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True  # Keep classification head trainable
    
    return model


class FFRClassifier(nn.Module):
    def __init__(self, threshold=0.75):
        super(FFRClassifier, self).__init__()
        self.threshold = threshold
        self.max =  1.0
        self.min =  0.20
        self.normalized_threshold = (self.threshold - self.min) / (self.max - self.min)

    def forward(self, ffr_value):
        return (ffr_value <= self.normalized_threshold).float()


class DSClassifier(nn.Module):
    def __init__(self, threshold=50.0):
        super(DSClassifier, self).__init__()
        self.threshold = threshold
        self.max =  98.3
        self.min =  7.78
        self.normalized_threshold = (self.threshold - self.min) / (self.max - self.min)

    def forward(self, ds_value):
        return (ds_value >= self.normalized_threshold).float()


def vit_large_patch16(use_metadata, freeze_backbone, no_visuals, metadata_len, **kwargs):
    if use_metadata:
        model = FusionVisionTransformer(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, freeze_backbone=freeze_backbone,
            no_visuals=no_visuals, metadata_len=metadata_len, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    else :
        model = VisionTransformer(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, freeze_backbone=freeze_backbone,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def resnet(use_metadata, freeze_backbone, metadata_len, **kwargs):
    model = models.resnet152(pretrained=True)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    return model

def ffr_classifier(use_metadata, freeze_backbone, metadata_len, **kwargs):
    assert use_metadata==True
    
    model = FFRClassifier()
    return model

def ds_classifier(use_metadata, freeze_backbone, metadata_len, **kwargs):
    assert use_metadata==True
    
    model = DSClassifier()
    return model


