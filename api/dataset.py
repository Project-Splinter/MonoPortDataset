import os
import numpy as np
from PIL import Image
import trimesh

import torch
import torchvision.transforms as transforms

from hoppeMesh import HoppeMesh
from sample import sample_surface

def projection(points, calib):
    return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]

class PIFuDataset():
    def __init__(self, opt, split='train', name='pifu_orth'):
        self.opt = opt
        self.split = split
        self.name = name
        self.projection_mode = 'orthogonal'
        self.input_size = 512
        
        self.root = opt.root # <Where-is-MonoPortDataset>/data/

        self.motion_list = sorted(
            np.loadtxt(os.path.join(self.root, self.name, f'{self.split}.txt'), dtype=str).tolist())
        self.rotations = range(0, 360, 10)
        
    def __len__(self):
        return len(self.motion_list) * len(self.rotations)

    def __getitem__(self, index):
        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        motion = self.motion_list[mid]
        subject, action, frame = motion[0], motion[1], int(motion[2])
        
        # setup paths
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

            'del_faces': np.load(os.path.join(
                self.root, 'renderppl', 'del_inside', subject, 'del_faces.npy')),
            'del_verts': np.load(os.path.join(
                self.root, 'renderppl', 'del_inside', subject, 'del_verts.npy')),
        }

        # load training data
        data_dict.update(self.load_calib(data_dict))
        data_dict.update(self.load_mesh(data_dict))
        # data_dict.update(self.load_image(data_dict))
        
        # # sampling
        # if self.opt.num_sample_geo:
        #     sample_data_geo = self.get_sampling_geo(data_dict)
        #     data_dict.update(sample_data_geo)

        # if self.opt.num_sample_color:
        #     sample_data_color = self.get_sampling_color(data_dict)
        #     data_dict.update(sample_data_color)

        return data_dict

    def load_image(self, data_dict):
        # return PIL.Image
        rgba = Image.open(data_dict['render_path']).convert('RGBA')
        mask = rgba.split()[-1]
        image = rgba.convert('RGB')
        return {'image': image, 'mask': mask}
    
    def load_calib(self, data_dict):
        # return numpy
        calib_data = np.loadtxt(data_dict['calib_path'], dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        intrinsic[1, :] *= -1
        calib_mat = np.matmul(intrinsic, extrinsic)
        return {'calib': calib_mat}

    def load_mesh(self, data_dict):
        mesh_ori = trimesh.load(data_dict['mesh_path'])
        
        verts = mesh_ori.vertices
        vert_normals = mesh_ori.vertex_normals
        face_normals = mesh_ori.face_normals
        faces = mesh_ori.faces

        if self.opt.num_sample_color:
            uvs = mesh_ori.visual.uv
            texture = Image.open(data_dict['uvrender_path'])
        else:
            uvs = None
            texture = None

        mesh = HoppeMesh(
            verts, vert_normals, face_normals, faces, 
            uvs=uvs, texture=texture)
        return {'mesh': mesh}

    def get_sampling_geo(self, data_dict):
        # return numpy
        mesh = data_dict['mesh']

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        samples_surface, face_index = sample_surface(
            mesh.triangles(), n_samples_surface, ignore_face_idxs=data_dict['del_faces'])
        offset = np.random.normal(
            scale=self.opt.sigma_geo, size=(n_samples_surface, 1))
        samples_surface += mesh.face_normals[face_index] * offset
        
        # Uniform samples in [-1, 1]
        calib_inv = np.linalg.inv(data_dict['calib'])
        b_min = projection(np.array([-1.0, -1.0, -1.0]), calib_inv)
        b_max = projection(np.array([1.0, 1.0, 1.0]), calib_inv)
        n_samples_space = self.opt.num_sample_geo // 4
        samples_space = np.random.rand(n_samples_space, 3) * (b_max - b_min) + b_min
        
        # total sampled points
        samples = np.concatenate([samples_surface, samples_space], 0)
        np.random.shuffle(samples)

        # labels: in->1.0; out->0.0.
        inside = mesh.contains(samples)

        # balance in and out
        inside_samples = samples[inside > 0.5]
        outside_samples = samples[inside <= 0.5]

        nin = inside_samples.shape[0]
        if nin > self.opt.num_sample_geo // 2:
            inside_samples = inside_samples[:self.opt.num_sample_geo // 2]
            outside_samples = outside_samples[:self.opt.num_sample_geo // 2]
        else:
            outside_samples = outside_samples[:(self.opt.num_sample_geo - nin)]
            
        samples = np.concatenate([inside_samples, outside_samples], 0)
        labels = np.concatenate([
            np.ones(inside_samples.shape[0]), np.zeros(outside_samples.shape[0])])

        return {
            'samples_geo': samples, 
            'labels_geo': labels,
        }

    def get_sampling_color(self, data_dict):
        # return numpy
        mesh = data_dict['mesh']
        samples, face_index = sample_surface(
            mesh.triangles(), self.opt.num_sample_color, ignore_face_idxs=data_dict['del_faces'])
        colors = mesh.get_colors(samples, mesh.faces[face_index])

        # Samples are around the true surface with an offset
        offset = np.random.normal(
            0, self.opt.sigma_color, (self.opt.num_sample_color, 1))
        samples += mesh.face_normals[face_index] * offset

        # Normalized to [-1, 1]
        colors = ((colors[:, 0:3] / 255.0) - 0.5) / 0.5
        return {
            'samples_color': samples, 
            'labels_color': colors
        }

    def visualize_sampling(self, data_dict, save_dir, mode='geo'):
        assert mode in ['geo', 'color']

        # [-1, 1]
        points = projection(data_dict[f'samples_{mode}'], data_dict['calib'])
        image = np.array(data_dict['image']).copy()

        for coord in points:
            coord = coord / 2 + 0.5
            x = int(coord[0] * image.shape[1])
            y = int(coord[1] * image.shape[0])
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            image[y, x] = 128
        im = Image.fromarray(np.uint8(image))
        im.save(save_dir)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/')
    parser.add_argument('--num_sample_geo', type=int, default=5000)
    parser.add_argument('--num_sample_color', type=int, default=0)
    parser.add_argument('--sigma_geo', type=float, default=0.05)
    parser.add_argument('--sigma_color', type=float, default=0.001)
    args = parser.parse_args()
        
    dataset = PIFuDataset(args, split='debug')
    data_dict = dataset[0]

    # dataset.visualize_sampling(data_dict, '../test_data/proj_geo.jpg', mode='geo')
    # dataset.visualize_sampling(data_dict, '../test_data/proj_color.jpg', mode='color')

    ## speed 3.30 iter/s
    import tqdm
    for _ in tqdm.tqdm(dataset):
        pass

