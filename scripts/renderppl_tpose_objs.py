# This is a blender python script.

import os
import sys
import argparse
import random
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/'))

from renderppl_mixamo_blender_tool import RenderpplBlenderTool, blender_print

tool = RenderpplBlenderTool()

subjects = np.loadtxt('../data/renderppl/all.txt', dtype=str)
np.random.shuffle(subjects)

for i, subject in enumerate(subjects):
    blender_print(subject, f'{i}/{len(subjects)}')
    subject_file = f'../data/renderppl/rigged/{subject}_FBX/{subject}_u3d.fbx'
    export_file = f'../data/renderppl/tpose_objs/{subject}.obj'
    
    if os.path.exists(export_file):
        blender_print(subject, 'skip because of exist!')
        continue
    os.makedirs(os.path.dirname(export_file), exist_ok=True)

    children_id = 0
    # special cases
    if subject in ["rp_adanna_rigged_004", "rp_emma_rigged_010"]:
        children_id = 1

    tool.import_model_fbx(subject_file, subject, children_id)
    tool.export_mesh(export_file)