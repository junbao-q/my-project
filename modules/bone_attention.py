import torch
import torch.nn as nn

class BoneCrossAttention(nn.Module):
  
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim **-0.5
        
        self.qkv_joint = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.qkv_bone = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        
    def forward(self, x_joint, x_bone):
        
        B, T, J, C = x_joint.shape
        _, _, B_num, _ = x_bone.shape
       
        qkv_joint = self.qkv_joint(x_joint).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q_j, k_j, v_j = qkv_joint[0], qkv_joint[1], qkv_joint[2]
        
       
        qkv_bone = self.qkv_bone(x_bone).reshape(B, T, B_num, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q_b, k_b, v_b = qkv_bone[0], qkv_bone[1], qkv_bone[2]
        
        if self.mode == 'spatial':
            
            attn_joint = (q_j @ k_j.transpose(-2, -1)) * self.scale
            attn_joint = attn_joint.softmax(dim=-1)
            attn_joint = self.attn_drop(attn_joint)
            
            attn_bone = (q_j @ k_b.transpose(-2, -1)) * self.scale
            attn_bone = attn_bone.softmax(dim=-1)
            attn_bone = self.attn_drop(attn_bone)
            
            x = attn_joint @ v_j + attn_bone @ v_b
            x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C)
            
        elif self.mode == 'temporal':
          
            q_j = q_j.transpose(2, 3)  # [B, H, J, T, C]
            k_j = k_j.transpose(2, 3)
            v_j = v_j.transpose(2, 3)
            
            q_b = q_b.transpose(2, 3)
            k_b = k_b.transpose(2, 3)
            v_b = v_b.transpose(2, 3)
            
            attn_joint = (q_j @ k_j.transpose(-2, -1)) * self.scale
            attn_joint = attn_joint.softmax(dim=-1)
            attn_joint = self.attn_drop(attn_joint)
            
            attn_bone = (q_j @ k_b.transpose(-2, -1)) * self.scale
            attn_bone = attn_bone.softmax(dim=-1)
            attn_bone = self.attn_drop(attn_bone)
            
            x = attn_joint @ v_j + attn_bone @ v_b
            x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x