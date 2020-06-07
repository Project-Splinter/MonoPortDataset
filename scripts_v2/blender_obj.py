# This is a blender python script.

import os
import sys
import argparse
import random
import glob
import math
import time
import json
import numpy as np
from collections import OrderedDict

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/'))

from renderppl_mixamo_blender_tool import RenderpplBlenderTool, blender_print

data = {
    'mixamo': {
        'train': json.load(open('../data/mixamo/clusters/kmeans/train_100.json', 'r')),
        'val': json.load(open('../data/mixamo/clusters/kmeans/val_10.json', 'r')),},
    'renderppl': {
        'train': np.loadtxt('../data/renderppl/train.txt', dtype=str),
        'val': np.loadtxt('../data/renderppl/val.txt', dtype=str),}
}

def random_motion(subject, split):
    actions = data['mixamo'][split]
    ncluster = len(actions.keys())
    
    icluster = random.randint(0, ncluster-1)
    action, frame = random.choice(actions[f'{icluster}'])
    return action, int(frame)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--subject', type=str, help='renderppl subject name')
parser.add_argument(    
    '-o', '--out_dir', type=str, default='/home/rui/mnt/dalong-win/pifu_orth_v2/', help='output save dir')
parser.add_argument(    
    '-n', '--num', type=int, default=1, help='number of objs (motion) per subject')
parser.add_argument(    
    '--split', type=str, default='train', choices=['train', 'val'])

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

subject = args.subject
save_folder = args.out_dir
motion_per_subject = args.num
split = args.split

# blender tool
tool = RenderpplBlenderTool()

# load renderppl subject
subject_file = f'../data/renderppl/rigged/{subject}_FBX/{subject}_u3d.fbx'
children_id = 0
if subject in ["rp_adanna_rigged_004", "rp_emma_rigged_010"]: # special cases
    children_id = 1
tool.import_model_fbx(subject_file, subject, children_id)

# start exporting objs
for _ in range(motion_per_subject):
    action, frame = random_motion(subject, split)

    # load mixamo action
    action_file = f'../data/mixamo/actions/{action}.fbx'
    tool.import_action_fbx(action_file, action)
    tool.apply_action(action)
    tool.set_frame(frame)

    # export obj
    export_obj_file = os.path.join(
        save_folder, subject, action, f'{frame:06d}', 'mesh.obj')
    os.makedirs(os.path.dirname(export_obj_file), exist_ok=True)
    tool.export_mesh(export_obj_file)

    # calculate prt
    excute_file = os.path.join(os.path.dirname(__file__), 'render_with_prt.py')
    os.system(f'python {excute_file} -s {subject} -a {action} -f {frame} -o {save_folder}')

    