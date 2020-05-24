import trimesh
import time
import tqdm
import numpy as np

mesh_file = "../data/pifu_orth/rp_adanna_rigged_005/Goalkeeper_Overhand_Throw/000086/mesh.obj"

for _ in tqdm.tqdm(range(1)): # 6 fps
    mesh_ori = trimesh.load(mesh_file)
    verts = mesh_ori.vertices
    vert_normals = mesh_ori.vertex_normals
    face_normals = mesh_ori.face_normals
    faces = mesh_ori.faces
    uvs = mesh_ori.visual.uv
    print (f"verts: {verts.shape}; faces: {faces.shape}; normals: {vert_normals.shape}; uvs: {uvs.shape}")

import tinyobjloader

def obj_loader(path):
   # Create reader.
    reader = tinyobjloader.ObjReader()

    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(path)

    if ret == False:
        print("Failed to load : ", path)
        return None

    # note here for wavefront obj, #v might not equal to #vt, same as #vn.
    attrib = reader.GetAttrib()
    v = np.array(attrib.vertices).reshape(-1, 3)
    vn = np.array(attrib.normals).reshape(-1, 3)
    vt = np.array(attrib.texcoords).reshape(-1, 2)

    shapes = reader.GetShapes()
    tri = shapes[0].mesh.numpy_indices().reshape(-1, 9)
    f_v = tri[:, [0, 3, 6]]
    f_vn = tri[:, [1, 4, 7]]
    f_vt = tri[:, [2, 5, 8]]
    
    faces = f_v #[m, 3]
    face_normals = vn[f_vn].mean(axis=1) #[m, 3]
    face_uvs = vt[f_vt].mean(axis=1) #[m, 2]

    verts = v #[n, 3]
    vert_normals = np.zeros((verts.shape[0], 3), dtype=np.float32) #[n, 3]
    vert_normals[f_v.reshape(-1)] = vn[f_vn.reshape(-1)]
    vert_uvs = np.zeros((verts.shape[0], 2), dtype=np.float32) #[n, 2]
    vert_uvs[f_v.reshape(-1)] = vt[f_vt.reshape(-1)]
    
    return verts, faces, vert_normals, face_normals, vert_uvs, face_uvs
    
for _ in tqdm.tqdm(range(50)):
    verts, faces, vert_normals, face_normals, vert_uvs, face_uvs = obj_loader(mesh_file)
    print (f"verts: {verts.shape}; faces: {faces.shape}; normals: {vert_normals.shape}; uvs: {uvs.shape}")