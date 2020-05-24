import os
import sys
import numpy as np
from PIL import Image
import trimesh
import glob
from collections import defaultdict

import torch
import torchvision.transforms as transforms

from .hoppeMesh import HoppeMesh
from .sample import sample_surface
from .mesh_util import obj_loader

def projection(points, calib):
    return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]

class PIFuDataset():
    def __init__(self, opt, split='train', name='pifu_orth'):
        self.opt = opt
        self.split = split
        self.name = name
        self.projection_mode = 'orthogonal'
        self.input_size = 512
        # current data should be loaded only by trimesh
        self.use_trimesh = False 
        # <Where-is-MonoPortDataset>/data/
        self.root = opt.root 

        self.motion_list = self.get_motion_list(split)
        self.rotations = range(0, 360, 10)

        # PIL to tensor
        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(
                brightness=opt.aug_bri, 
                contrast=opt.aug_con, 
                saturation=opt.aug_sat,
                hue=opt.aug_hue)
        ])


    def get_motion_list(self, split):
        txt = os.path.join(self.root, self.name, f'{split}.txt')
        if os.path.exists(txt):
            print (f"load from {txt}")
            motion_list = sorted(np.loadtxt(txt, dtype=str).tolist())
        else:
            print (f"load the entire dataset and exclude val & test set.")
            # load the val/test list 
            val_txt = os.path.join(self.root, self.name, 'val.txt')
            test_txt = os.path.join(self.root, self.name, 'test.txt')
            assert os.path.exists(val_txt) or os.path.exists(test_txt)
            skip_list = []
            if os.path.exists(val_txt):
                skip_list += sorted(np.loadtxt(val_txt, dtype=str).tolist())
            if os.path.exists(test_txt):
                skip_list += sorted(np.loadtxt(test_txt, dtype=str).tolist())
            skip_list = ['_'.join(motion) for motion in skip_list]
            
            # scan the entire folder and ignore this list in val/test
            paths = sorted(glob.glob(os.path.join(self.root, self.name, '*/*/*/render')))
            motion_list = []
            for path in paths:
                splits = path.split('/')
                motion = [splits[-4], splits[-3], str(int(splits[-2]))]
                if '_'.join(motion) in skip_list:
                    continue
                motion_list.append(motion)

        return motion_list
        
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
        }

        # load preprocessed data for eyebow & teeth removal
        if self.use_trimesh:
            data_dict.update({
                'del_faces': np.load(os.path.join(
                    self.root, 'renderppl', 'del_inside', subject, 'del_faces.npy')),
                'del_verts': np.load(os.path.join(
                    self.root, 'renderppl', 'del_inside', subject, 'del_verts.npy')),
            })
        else:
            data_dict.update({
                'del_faces': np.load(os.path.join(
                    self.root, 'renderppl', 'del_inside_raw', subject, 'del_faces.npy')),
                'del_verts': np.load(os.path.join(
                    self.root, 'renderppl', 'del_inside_raw', subject, 'del_verts.npy')),
            })

        # load training data
        data_dict.update(self.load_calib(data_dict))
        data_dict.update(self.load_mesh(data_dict))
        data_dict.update(self.load_image(data_dict))

        assert data_dict['del_faces'].shape[0] == data_dict['mesh'].faces.shape[0]
        assert data_dict['del_verts'].shape[0] == data_dict['mesh'].verts.shape[0]
        
        # sampling
        if self.opt.num_sample_geo:
            sample_data_geo = self.get_sampling_geo(data_dict)
            data_dict.update(sample_data_geo)

        if self.opt.num_sample_color:
            sample_data_color = self.get_sampling_color(data_dict)
            data_dict.update(sample_data_color)

        del data_dict['del_faces']
        del data_dict['del_verts']
        del data_dict['mesh']

        return data_dict

    def load_image(self, data_dict):
        # return PIL.Image
        rgba = Image.open(data_dict['render_path']).convert('RGBA')
        mask = rgba.split()[-1]
        image = rgba.convert('RGB')

        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)
        image = image * mask
        return {'image': image, 'mask': mask}
    
    def load_calib(self, data_dict):
        calib_data = np.loadtxt(data_dict['calib_path'], dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        intrinsic[1, :] *= -1
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_mat = torch.from_numpy(calib_mat)
        return {'calib': calib_mat}

    def load_mesh(self, data_dict):
        mesh_path = data_dict['mesh_path']
        if self.opt.num_sample_color:            
            texture = Image.open(data_dict['uvrender_path'])
        else:
            texture = None
        if 'del_verts' in data_dict:
            ignore_vert_idxs = data_dict['del_verts']
        else:
            ignore_vert_idxs = None
        if 'del_faces' in data_dict:
            ignore_face_idxs = data_dict['del_faces']
        else:
            ignore_face_idxs = None

        if self.use_trimesh:
            mesh_ori = trimesh.load(mesh_path)
            verts = mesh_ori.vertices
            vert_normals = mesh_ori.vertex_normals
            face_normals = mesh_ori.face_normals
            faces = mesh_ori.faces
            vert_uvs = mesh_ori.visual.uv
        else:
            # use tinyobjloader here, faster
            verts, faces, vert_normals, face_normals, \
                vert_uvs, face_uvs = obj_loader(mesh_path)

        mesh = HoppeMesh(
            verts, faces, vert_normals, face_normals, 
            uvs=vert_uvs, texture=texture,
            ignore_vert_idxs=ignore_vert_idxs,
            ignore_face_idxs=ignore_face_idxs)
        return {'mesh': mesh}

    def get_sampling_geo(self, data_dict):
        mesh = data_dict['mesh']
        calib = data_dict['calib']
        if 'del_faces' in data_dict:
            ignore_face_idxs = data_dict['del_faces']
        else:
            ignore_face_idxs = None

        # Samples are around the true surface with an offset
        n_samples_surface = 4 * self.opt.num_sample_geo
        samples_surface, face_index = sample_surface(
            mesh.triangles(), n_samples_surface, ignore_face_idxs=ignore_face_idxs)
        offset = np.random.normal(
            scale=self.opt.sigma_geo, size=(n_samples_surface, 1))
        samples_surface += mesh.face_normals[face_index] * offset
        
        # Uniform samples in [-1, 1]
        calib_inv = np.linalg.inv(calib)
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

        samples = torch.from_numpy(samples)
        labels = torch.from_numpy(labels)
        return {
            'samples_geo': samples, 
            'labels_geo': labels,
        }

    def get_sampling_color(self, data_dict):
        mesh = data_dict['mesh']
        if 'del_faces' in data_dict:
            ignore_face_idxs = data_dict['del_faces']
        else:
            ignore_face_idxs = None

        samples, face_index = sample_surface(
            mesh.triangles(), self.opt.num_sample_color, ignore_face_idxs=ignore_face_idxs)
        colors = mesh.get_colors(samples, mesh.faces[face_index])

        # Samples are around the true surface with an offset
        offset = np.random.normal(
            0, self.opt.sigma_color, (self.opt.num_sample_color, 1))
        samples += mesh.face_normals[face_index] * offset

        # Normalized to [-1, 1] rgb
        colors = ((colors[:, 0:3] / 255.0) - 0.5) / 0.5

        samples = torch.from_numpy(samples)
        colors = torch.from_numpy(colors)
        return {
            'samples_color': samples, 
            'labels_color': colors
        }

    def visualize_sampling(self, data_dict, save_dir, mode='geo'):
        assert mode in ['geo', 'color']
        samples = data_dict[f'samples_{mode}'].numpy()
        labels = data_dict[f'labels_{mode}'].numpy()
        if mode == 'geo':
            colors = np.stack([labels, labels, labels], axis=1)
        else:
            colors = labels * 0.5 + 0.5
            print(colors.max())
        image = data_dict['image'].numpy().transpose(1, 2, 0)
        calib = data_dict['calib'].numpy()

        # [-1, 1]
        points = projection(samples, calib)
        image = np.array(image).copy()

        for i, coord in enumerate(points):
            coord = coord / 2 + 0.5
            x = int(coord[0] * image.shape[1])
            y = int(coord[1] * image.shape[0])
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            image[y, x] = np.uint8(colors[i] * 255) #128
        im = Image.fromarray(np.uint8(image))
        im.save(save_dir)

    def visualize_sampling3D(self, data_dict, mode='geo'):
        import vtkplotter
        assert mode in ['geo', 'color']
        samples = data_dict[f'samples_{mode}'].numpy()
        labels = data_dict[f'labels_{mode}'].numpy()
        if mode == 'geo':
            colors = np.stack([labels, labels, labels], axis=1)
        else:
            colors = labels * 0.5 + 0.5

        mesh = data_dict['mesh']
        calib = data_dict['calib'].numpy()

        if 'del_faces' in data_dict:
            ignore_face_idxs = data_dict['del_faces']
            faces = mesh.faces[~ignore_face_idxs]
        else:
            faces = mesh.faces

        # [-1, 1]
        points = projection(samples, calib)
        verts = projection(mesh.verts, calib)
        
        # create plot
        vp = vtkplotter.Plotter(title="", size=(1500, 1500))
        vis_list = []

        # create a mesh
        mesh = trimesh.Trimesh(verts, faces)
        mesh.visual.face_colors = [200, 200, 250, 255]
        vis_list.append(mesh)

        # create a pointcloud
        pc = vtkplotter.Points(points, r=12, c=np.float32(colors))
        vis_list.append(pc)
        
        vp.show(*vis_list, bg="white", axes=1, interactive=True)