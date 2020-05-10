# This is a blender python script.

import os
import sys
import argparse
import random
import glob
import math
import time
import numpy as np
from collections import OrderedDict

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/'))

from renderppl_mixamo_blender_tool import RenderpplBlenderTool, blender_print

# -------- settings ----------
# random.seed(42)
save_folder = '../data/pifu_orth/'
splits = ['val', 'train']
resolution = 512
uv_resolution = 512
num_samples = 100 # seems like 100 is enough
use_motion_blur = False
use_transparent_bg = True
use_denoising = True
# -----------------------------

tool = RenderpplBlenderTool()
tool.set_resolution(resolution, resolution)
tool.set_render(
    num_samples=num_samples, 
    use_motion_blur=use_motion_blur, 
    use_transparent_bg=use_transparent_bg, 
    use_denoising=use_denoising)


for split in splits:
    motions = np.loadtxt(os.path.join(save_folder, f'{split}.txt'), dtype=str)
    
    # re-organizing the processing list so that we won't load same ppl/action twice.
    processing_dict = {}
    for motion in motions:
        subject, action, frame = motion[0], motion[1], int(motion[2])
        if subject not in processing_dict:
            processing_dict[subject] = {}
        if action not in processing_dict[subject]:
            processing_dict[subject][action] = []
        processing_dict[subject][action].append(frame)
    
    # for multi processing
    processing_items = list(processing_dict.items())
    random.shuffle(processing_items)
    processing_dict = OrderedDict(processing_items)

    for i, (subject, subject_dict) in enumerate(processing_dict.items()):
        tic = time.time()
        blender_print (f'[{i}/{len(processing_dict.items())}]', subject)
        # load renderppl subject
        subject_file = f'../data/renderppl/rigged/{subject}_FBX/{subject}_u3d.fbx'
        tex_file = f'../data/renderppl/rigged/{subject}_FBX/tex/{subject}_dif.jpg'
        normal_file = f'../data/renderppl/rigged/{subject}_FBX/tex/{subject}_norm.jpg'
        tool.import_model_fbx(subject_file, subject)
        tool.import_material(
            tex_file, normal_file, 
            Specular=0.1, Metallic=0.0, Transmission=0.0, Anisotropic=0.0)
        
        check_path = os.path.join(save_folder, subject)
        if os.path.exists(check_path):
            blender_print ('skip', subject)
            continue
        
        for j, (action, action_list) in enumerate(subject_dict.items()):
            blender_print (
                f'[{i}/{len(processing_dict.items())}]', subject,
                f'[{j}/{len(subject_dict.items())}]', action)
            # load mixamo action
            action_file = f'../data/mixamo/actions/{action}.fbx'
            tool.import_action_fbx(action_file, action)
            tool.apply_action(action)

            for frame in action_list:
                # set frame
                tool.set_frame(frame)
                # random a lighting env
                hdri_file = random.choice(glob.glob(f'../data/hdri/*.exr'))
                tool.import_world_lighting(hdri_file)
                
                # export obj
                export_obj_file = os.path.join(
                    save_folder, subject, action, f'{frame:06d}', 'mesh.obj')
                os.makedirs(os.path.dirname(export_obj_file), exist_ok=True)
                tool.export_mesh(export_obj_file)

                # export skeleton
                export_sk_file = os.path.join(
                    save_folder, subject, action, f'{frame:06d}', 'skeleton.txt')
                os.makedirs(os.path.dirname(export_sk_file), exist_ok=True)
                tool.export_skeleton(export_sk_file)

                # export uv render
                export_uvrender_file = os.path.join(
                    save_folder, subject, action, f'{frame:06d}', 'uv_render.png')
                os.makedirs(os.path.dirname(export_uvrender_file), exist_ok=True)
                tool.set_uv_render(uv_resolution, uv_resolution)
                tool.render_to_uv(export_uvrender_file)

                # camera settings.
                skeleton = np.loadtxt(export_sk_file, dtype=float, usecols=[1,2,3])
                skeleton_names = np.loadtxt(export_sk_file, dtype=str, usecols=[0]).tolist()
                center = skeleton[skeleton_names.index('hip'), :] / 100
                dist, near, far = 3, 1.5, 4.5
                ortho_scale = 2.00
                rotations = range(0, 360, 36) if split=='train' else range(0, 360, 120)
                for elev in rotations:
                    # set camera: this sets a camera looking at {lookat#0} from {dist#1}
                    # far, positioned at {rad#2} degree between x axis, and at the 
                    # same height as lookat.
                    tool.set_camera_ortho_pifu(
                        center, dist, math.radians(elev), near, far, ortho_scale)
                    
                    # export rendered image and calib matrix
                    export_render_file = os.path.join(
                        save_folder, subject, action, 
                        f'{frame:06d}', 'render', f'{elev:03d}.png')
                    os.makedirs(os.path.dirname(export_render_file), exist_ok=True)
                    export_calib_file = os.path.join(
                        save_folder, subject, action, 
                        f'{frame:06d}', 'calib', f'{elev:03d}.txt')
                    os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
                    tool.render_to_img(export_render_file, export_calib_file)

                break
            
            del tool.action_pool[action]

        tool.reset()
        tool.init_camera()
        tool.set_resolution(resolution, resolution)
        tool.set_render(
            num_samples=num_samples, 
            use_motion_blur=use_motion_blur, 
            use_transparent_bg=use_transparent_bg, 
            use_denoising=use_denoising)

            
        toc = time.time()
        blender_print(f'{subject} finished! It takes {(toc-tic)/60:.3f} min')   

blender_print('done.')
            

