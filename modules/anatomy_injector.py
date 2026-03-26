import torch
import torch.nn as nn
from timm.models.layers import DropPath

from model.modules.attention import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP
from model.modules.bone_attention import BoneCrossAttention


class AnatomyInjectorBlock(nn.Module):
  
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial'):
        super().__init__()
        self.norm1_joint = nn.LayerNorm(dim)
        self.norm1_bone = nn.LayerNorm(dim)
        self.norm1_limb = nn.LayerNorm(dim)
        
        
        self.bone_cross_attn = BoneCrossAttention(
            dim_in=dim, dim_out=dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, mode=mode
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            
    def forward(self, x_joint, x_bone, x_limb):
        
        if self.use_layer_scale:
          
            x_joint = x_joint + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) * 
                self.bone_cross_attn(self.norm1_joint(x_joint), self.norm1_bone(x_bone))
            )
          
            x_joint = x_joint + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) * 
                self.mlp(self.norm2(x_joint))
            )
        else:
            x_joint = x_joint + self.drop_path(
                self.bone_cross_attn(self.norm1_joint(x_joint), self.norm1_bone(x_bone))
            )
            x_joint = x_joint + self.drop_path(
                self.mlp(self.norm2(x_joint))
            )
        return x_joint


class AnatomyInjector(nn.Module):
    ""
    def __init__(self, dim, num_layers=2, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
       
        self.spatial_injector = AnatomyInjectorBlock(
            dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, attn_drop=attn_drop,
            drop=drop, drop_path=drop_path, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value, mode='spatial'
        )
        
        self.temporal_injector = AnatomyInjectorBlock(
            dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, attn_drop=attn_drop,
            drop=drop, drop_path=drop_path, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value, mode='temporal'
        )
        
        
        self.fusion = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x_joint, x_bone, x_limb):
      
        spatial_out = self.spatial_injector(x_joint, x_bone, x_limb)
    
        temporal_out = self.temporal_injector(x_joint, x_bone, x_limb)
        
        
        fused = torch.cat([spatial_out, temporal_out], dim=-1)
        fused = self.fusion(fused)
        return self.norm(fused)