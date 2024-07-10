#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, normal, mask, gt_alpha_mask,
                 image_name, normal_name, depth_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", panorama=False
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.normal_name = normal_name
        self.depth_name = depth_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.raw_mask = mask
        self.is_masked = None
        if mask is not None:
            self.is_masked = (mask == 0).expand(*image.shape)  # True represent masked pixel
        """
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        """
        self.depth = None
        if depth is not None:
            self.depth = depth.clamp(0.0, 1.0).to(self.data_device)

        self.normal = None
        if normal is not None:
            self.normal = normal.clamp(0.0, 1.0).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.original_image = None
        self.normal = None
        self.depth = None

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.panorama = panorama
        if panorama:
            #self.image_width = self.original_image.shape[2] // 4
            #self.oritinal_image_width = self.original_image.shape[2] # for low resolution

            #self.image_height = self.original_image.shape[1]
            R_r, T_r = self.rotate_camera_coordinate(R, T)

            self.world_view_transform_right = torch.tensor(getWorld2View2(R_r, T_r, trans, scale)).transpose(0, 1).cuda()
            self.full_proj_transform_right = (self.world_view_transform_right.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center_right = self.world_view_transform_right.inverse()[3, :3]
            R_b, T_b = self.rotate_camera_coordinate(R_r, T_r)

            self.world_view_transform_back = torch.tensor(getWorld2View2(R_b, T_b, trans, scale)).transpose(0, 1).cuda()
            self.full_proj_transform_back = (self.world_view_transform_back.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center_back = self.world_view_transform_back.inverse()[3, :3]

            R_l, T_l = self.rotate_camera_coordinate(R_b, T_b)
            self.world_view_transform_left = torch.tensor(getWorld2View2(R_l, T_l, trans, scale)).transpose(0, 1).cuda()
            self.full_proj_transform_left = (self.world_view_transform_left.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center_left = self.world_view_transform_left.inverse()[3, :3]



    def rotate_camera_coordinate(self, R, T):
        R_y = np.array([[ 0.0, 0.0,  1.0, 0.0], [ 0.0,  1.0,  0.0, 0.0], [ -1.0,  0.0,  0.0, 0.0], [ 0.0,  0.0,  0.0, 1.0]])

        Rt = np.zeros((4, 4)) # w2c
        Rt[:3, :3] = R.transpose() 
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0  
        c2w_tmp = np.linalg.inv(Rt) # c2w
        RT_c2w = np.matmul(c2w_tmp,R_y.transpose())
        R = RT_c2w[:3, :3]
        T = np.linalg.inv(RT_c2w)[:3, 3]
        return R, T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

