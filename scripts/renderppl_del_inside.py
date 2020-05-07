import numpy as np
import os
import trimesh
import tqdm
import multiprocessing

def find_teeth_comps(ori_mesh):
    components = trimesh.Trimesh(
        ori_mesh.vertices, ori_mesh.faces
    ).split(only_watertight=False)
    
    comps = []
    for comp in components:
        nv = comp.vertices.shape[0]
        if nv in [520, 427]:
            comps.append(comp)
    
    if len(comps) == 2:
        return comps
    
    else:
        print ("ERROR! nv:", [c.vertices.shape[0] for c in components])
        return None
    
def find_mouth_comps(ori_mesh, teeth_comps):
    components = ori_mesh.split(only_watertight=False)
    
    comps = []
    for teeth_comp in teeth_comps:
        surface_pts, _ = trimesh.sample.sample_surface_even(teeth_comp, 500)
        match_dist = 9999,
        match_comp = None
        for comp in components:
            mask = np.isin(comp.vertices, teeth_comp.vertices)
            mask = mask.sum(axis=1) == 3
            if mask.sum() > 0: # teeth comp
                continue
            
            _, dist, _ = trimesh.proximity.closest_point(comp, surface_pts)
            dist = dist.mean()
            if dist < match_dist:
                match_dist = dist
                match_comp = comp
        comps.append(match_comp)
#         print (match_dist, match_comp.vertices.shape[0], match_comp.faces.shape[0])
    return comps

def find_del_verts(ori_mesh, mouth_comps):
    components = trimesh.Trimesh(
        ori_mesh.vertices, ori_mesh.faces
    ).split(only_watertight=False)
    
    comp_num = [m.vertices.shape[0] for m in components]
    main_comp_num = max(comp_num)
    
    del_comps = mouth_comps
    for comp in components:
        if comp.vertices.shape[0] == main_comp_num:
            continue
        del_comps += [comp]
    
#     print (len(del_comps), [m.vertices.shape[0] for m in del_comps])
    
    del_vertices = np.concatenate([comp.vertices for comp in del_comps], axis=0)
    del_verts = np.isin(ori_mesh.vertices, del_vertices)
    del_verts = del_verts.sum(axis=1) == 3
    return del_verts

def find_del_faces(ori_mesh, del_verts):
    idxs_del = np.where(del_verts)[0]
    del_faces = np.isin(ori_mesh.faces, idxs_del)
    del_faces = del_faces.sum(axis=1) > 0
    
#     print (ori_mesh.faces.min())
    return del_faces

def process(subject, check=False):
    print (subject)
    obj_file = f'../data/renderppl/tpose_objs/{subject}.obj'
    ori_mesh = trimesh.load(obj_file)
    teeth_comps = find_teeth_comps(ori_mesh)
    if check:
        for i, comp in enumerate(teeth_comps):
            cache_file = os.path.join('../test_data', 'teeth_comps', f'{subject}_comp{i}.obj')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            comp.export(cache_file)
    
    mouth_comps = find_mouth_comps(ori_mesh, teeth_comps)
    if check:
        for i, comp in enumerate(mouth_comps):
            cache_file = os.path.join('../test_data', 'mouth_comps', f'{subject}_comp{i}.obj')
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            comp.export(cache_file)
    
    del_verts = find_del_verts(ori_mesh, mouth_comps)
    del_faces = find_del_faces(ori_mesh, del_verts)
    
    # save final
    final_file = f'../data/renderppl/del_inside/{subject}/clear.obj'
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    trimesh.Trimesh(
        ori_mesh.vertices, ori_mesh.faces[~del_faces]
    ).export(final_file)
    
    final_file = f'../data/renderppl/del_inside/{subject}/del_verts.npy'
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    np.save(final_file, del_verts)
    
    final_file = f'../data/renderppl/del_inside/{subject}/del_faces.npy'
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    np.save(final_file, del_faces)
    
subjects = np.loadtxt('../data/renderppl/all.txt', dtype=str)
pool = multiprocessing.Pool(8)
pool.map(process, subjects)