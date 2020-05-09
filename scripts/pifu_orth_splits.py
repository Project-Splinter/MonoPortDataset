# In {train, val}.txt
# each line sould be: <subject> <action> <frame>

import os
import json
import tqdm
import random
import numpy as np

random.seed(42)

data = {
    'mixamo': {
        'train': json.load(open('../data/mixamo/clusters/kmeans/train_100.json', 'r')),
        'val': json.load(open('../data/mixamo/clusters/kmeans/val_10.json', 'r')),},
    'renderppl': {
        'train': np.loadtxt('../data/renderppl/train.txt', dtype=str),
        'val': np.loadtxt('../data/renderppl/val.txt', dtype=str),}
}
motion_per_subject = 30

for split in ['train', 'val']:
    subjects = data['renderppl'][split]
    actions = data['mixamo'][split]
    ncluster = len(actions.keys())
    
    export_file = f'../data/pifu_orth/{split}.txt'
    os.makedirs(os.path.dirname(export_file), exist_ok=True)

    with open(export_file, 'w') as f:
        for subject in tqdm.tqdm(subjects):
            # random motions for #motion_per_subject
            motions = []
            for _ in range(motion_per_subject):
                icluster = random.randint(0, ncluster-1)
                action, frame = random.choice(actions[f'{icluster}'])
                motions.append([action, frame])
            motions = np.array(motions, dtype=str)
            motions = motions[motions[:, 0].argsort()]

            for motion in motions:
                action, frame = motion
                f.write(f'{subject} {action} {frame}\n')