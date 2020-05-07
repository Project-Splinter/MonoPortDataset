# This is a blender python script.

import os
import sys
import argparse
import random
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/'))

from renderppl_mixamo_blender_tool import RenderpplBlenderTool, blender_print

# init blender with a template subject
template_subject = '../data/renderppl/rigged/rp_yumiko_rigged_001_FBX/rp_yumiko_rigged_001_u3d.fbx'
tool = RenderpplBlenderTool()
tool.import_model_fbx(template_subject)

actions = np.loadtxt('../data/mixamo/all.txt', dtype=str)
np.random.shuffle(actions)

for i, action in enumerate(actions):
    blender_print(action, f'{i}/{len(actions)}')
    action_file = f'../data/mixamo/actions/{action}.fbx'
    
    # if exist, skip for saving time
    check_export_file = f'../data/mixamo/skeletons/{action}/{1:06d}.sk'
    if os.path.exists(check_export_file):
        blender_print(action, 'skip because of exist!')
        continue

    # import action
    tool.import_action_fbx(action_file, action)
    tool.apply_action(action)

    # forloop: frames
    duration = tool.get_action_duration(action)
    for frame in range(1, duration+1):
        export_file = f'../data/mixamo/skeletons/{action}/{frame:06d}.sk'
        os.makedirs(os.path.dirname(export_file), exist_ok=True)
        tool.set_frame(frame)
        tool.export_skeleton(export_file)
    del tool.action_pool[action]
