import torch
import torch.nn as nn
from utils.paramUtil import t2m_kinematic_chain

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
    
    def forward_joints(self, joints_pred, joints_gt):
        loss = self.Loss(joints_pred, joints_gt)
        return loss
    
    def forward_skeleton(self, joints_pred, joints_gt):
        bone_index_list = [(sublist[i], sublist[i+1]) for sublist in t2m_kinematic_chain for i in range(len(sublist)-1)]
        bone_starts = [i[0] for i in bone_index_list]
        bone_ends = [i[1] for i in bone_index_list]
        diff_pred = joints_pred[:, bone_starts, :] - joints_pred[:, bone_ends, :]
        bone_length_pred = torch.sqrt(torch.sum(diff_pred ** 2, dim=-1))
        diff_gt = joints_gt[:, bone_starts, :] - joints_gt[:, bone_ends, :]
        bone_length_gt = torch.sqrt(torch.sum(diff_gt ** 2, dim=-1))
        loss = self.Loss(bone_length_pred, bone_length_gt)
        return loss