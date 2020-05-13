import os
import numpy as np

import torch
import torchvision.transforms as transforms

class PIFuDataset():
    def __init__(self, opt, split='train', name='pifu_orth'):
        self.opt = opt
        self.split = split
        self.name = name
        self.projection_mode = 'orthogonal'
        
        self.root = opt.root # <Where-is-MonoPortDataset>/data/

        self.motion_list = self.get_motion_list()
        self.rotations = range(0, 360, 10)
        
    def get_motion_list(self):
        motion_list = np.loadtxt(
            os.path.join(self.root, self.name, f'{self.split}.txt'), dtype=str)
        return sorted(motion_list.tolist())

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
                self.root, self.name, subject, action, f'{frame:06d}', 'mesh.obj'),
            'skeleton_path': os.path.join(
                self.root, self.name, subject, action, f'{frame:06d}', 'skeleton.txt'),
            'uvrender_path': os.path.join(
                self.root, self.name, subject, action, f'{frame:06d}', 'uv_render.png'),
            'calib_path': os.path.join(
                self.root, self.name, subject, action, f'{frame:06d}', 'calib', f'{rotation:03d}.txt'),
            'render_path': os.path.join(
                self.root, self.name, subject, action, f'{frame:06d}', 'render', f'{rotation:03d}.png'),

            'del_faces_path': os.path.join(
                self.root, 'renderppl', 'del_inside', subject, 'del_faces.npy'),
            'del_verts_path': os.path.join(
                self.root, 'renderppl', 'del_inside', subject, 'del_verts.npy'),
        }
        for path in [
            'mesh_path', 'skeleton_path', 'uvrender_path', 'calib_path',
            'render_path', 'del_faces_path', 'del_verts_path']:
            assert os.path.exists(data_dict[path]), f'{data_dict[path]} not exist!'

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/')
    parser.add_argument('--num_sample_geo', type=int, default=5000)
    parser.add_argument('--num_sample_color', type=int, default=5000)
    args = parser.parse_args()
        
    dataset = PIFuDataset(args, split='debug')
    data = dataset[0]
    print (data)