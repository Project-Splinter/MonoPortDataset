import os
import numpy as np

import torch
import torchvision.transforms as transforms

from .hoppe_sdf import HoppeSDF

class PIFuDataset():
    def __init__(self, opt, split='train', name='pifu_orth'):
        self.opt = opt
        self.split = split
        self.name = name
        self.projection_mode = 'orthogonal'
        
        self.root = opt.root # <Where-is-MonoPortDataset>/

        self.motion_list = self.get_motion_list()
        self.rotations = range(0, 360, 36) if split=='train' else range(0, 360, 120)
        
    def get_motion_list(self):
        motion_list = np.loadtxt(
            os.path.join(self.root, self.name, f'{self.split}.txt'), dtype=str)
        return sorted(list(motion_list))

    def __len__(self):
        return len(self.motion_list) * len(self.rotations)

    def __getitem__(self, index):
        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        motion = self.motion_list[mid]
        subject, action, frame = motion[0], motion[1], int(motion[2])

        data_dict = {
            'subject': subject,
            'action': action,
            'frame': frame,
            'rotation': rotation,

            'mesh_path': os.path.join(
                self.root, self.name, subject, action, frame, 'mesh.obj'),
            'skeleton_path': os.path.join(
                self.root, self.name, subject, action, frame, 'skeleton.txt'),
            'uvrender_path': os.path.join(
                self.root, self.name, subject, action, frame, 'uv_render.png'),
            'uvpos_path': os.path.join(
                self.root, self.name, subject, action, frame, 'uv_pos.npy'),
            'calib_path': os.path.join(
                self.root, self.name, subject, action, frame, 'calib', f'{rotation:03d}.txt'),
            'calib_path': os.path.join(
                self.root, self.name, subject, action, frame, 'render', f'{rotation:03d}.png'),

            'del_faces_path': os.path.join(
                self.root, 'renderppl', 'del_inside', subject, 'del_faces.npy'),
            'del_verts_path': os.path.join(
                self.root, 'renderppl', 'del_inside', subject, 'del_verts.npy'),
            'del_uvmask_path': os.path.join(
                self.root, 'renderppl', 'del_inside', subject, 'del_uv_mask.png'),
        }

        if self.opt.num_sample_geo:
            sample_data_geo = self.get_sampling_geo(data_dict)
            data_dict.update(sample_data_geo)

        if self.opt.num_sample_color:
            sample_data_color = self.get_sampling_color(data_dict)
            data_dict.update(sample_data_color)

        return data_dict

    def get_sampling_geo(self, data_dict):
        return {}

    def get_sampling_color(self, data_dict):
        return {}

if __name__ == '__main__':
    pass
        
