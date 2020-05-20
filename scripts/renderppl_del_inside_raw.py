import sys
import numpy as np
import trimesh
import os
import tqdm
sys.path.insert(0, '../')

from api.mesh_util import load_obj_mesh

def find_idx_del_verts(verts, del_verts):
    idx_del_verts = np.isin(verts, del_verts)
    idx_del_verts = idx_del_verts.sum(axis=1) == 3
    return idx_del_verts

def find_idx_del_faces(faces, idx_del_verts):
    idxs_del = np.where(idx_del_verts)[0]
    idx_del_faces = np.isin(faces, idxs_del)
    idx_del_faces = idx_del_faces.sum(axis=1) > 0
    return idx_del_faces

subjects = np.loadtxt('../data/renderppl/all.txt', dtype=str)
for subject in tqdm.tqdm(subjects):
    print (subject)
    # trimesh load with change #verts
    obj_file = f'../data/renderppl/tpose_objs/{subject}.obj'
    mesh = trimesh.load(obj_file)
    
    path_del_verts = f'../data/renderppl/del_inside/{subject}/del_verts.npy'
    idx_del_verts = np.load(path_del_verts)
    del_verts = mesh.vertices[idx_del_verts]

    # a lightweight function that does not change anything.
    verts, faces, vert_norms, face_norms, uvs, face_uvs = load_obj_mesh(
        obj_file, with_normal=True, with_texture=True)

    idx_del_verts = find_idx_del_verts(verts, del_verts)
    idx_del_faces = find_idx_del_faces(faces, idx_del_verts)

    # save final
    final_file = f'../data/renderppl/del_inside_raw/{subject}/clear.obj'
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    trimesh.Trimesh(
        verts, faces[~idx_del_faces]
    ).export(final_file)
    
    final_file = f'../data/renderppl/del_inside_raw/{subject}/del_verts.npy'
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    np.save(final_file, idx_del_verts)
    
    final_file = f'../data/renderppl/del_inside_raw/{subject}/del_faces.npy'
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    np.save(final_file, idx_del_faces)
