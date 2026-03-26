import torch
import torch.nn as nn

def bone_decomposer(input_x: torch.Tensor) -> torch.Tensor:
    
    # input_x: [B, T, J, C]
    bone_child = [0,1,2, 0,4,5, 0,7,8,9, 8,11,12, 8,14,15]
    bone_parent = [1,2,3, 4,5,6, 7,8,9,10, 11,12,13, 14,15,16]
    
    
    bone_directions = input_x[:, :, bone_child] - input_x[:, :, bone_parent]
    
    bone_lengths = torch.norm(bone_directions, dim=-1, keepdim=True)
    bone_lengths = torch.clamp(bone_lengths, min=1e-8)  # 避免除零
    
    bone_directions = bone_directions / bone_lengths
    
   
    bone_directions_mean = torch.mean(bone_directions, dim=-2, keepdim=True)
    bone_lengths_mean = torch.mean(bone_lengths, dim=-2, keepdim=True)
    
    bone_directions = torch.cat((bone_directions, bone_directions_mean), dim=-2)
    bone_lengths = torch.cat((bone_lengths, bone_lengths_mean), dim=-2)
    
  
    bone_info = torch.cat((bone_directions, bone_lengths), dim=-1)
    return bone_info


class BoneRefusion(nn.Module):
    
    def __init__(self):
        super().__init__()
      
        self.symmetric_pairs = [
            (1, 4), (2, 5), (3, 6),  
            (11, 14), (12, 15), (13, 16),  
            (8, 8)  
        ]
        
    def forward(self, x):
        # x: [B, T, J, C]
        B, T, J, C = x.shape
        symmetric_features = []
        
        for (left, right) in self.symmetric_pairs:
          
            left_feat = x[:, :, left]
            right_feat = x[:, :, right]
            
       
            diff = left_feat - right_feat
            sum_feat = left_feat + right_feat
            ratio = torch.div(left_feat, right_feat + 1e-8)  
            
            symmetric_features.append(torch.cat([diff, sum_feat, ratio], dim=-1))
        
       
        symmetric_features = torch.stack(symmetric_features, dim=-2)
        return symmetric_features