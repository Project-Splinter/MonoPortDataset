#!/bin/bash
NUM_THREADS=$1

echo "Runing multiple blender instances ->" $NUM_THREADS

PYTHON_SCRIPT='./blender_obj.py'
BLENDER='../bin/blender-2.82a-linux64/blender'
SAVE_DIR='/media/external/data/pifu_orth_v0'
SPLIT='train'

cat ../data/renderppl/$SPLIT.txt | shuf | xargs -P$NUM_THREADS -I {} $BLENDER --background --python $PYTHON_SCRIPT -- --subject {} --split $SPLIT -o $SAVE_DIR -n 2 > /dev/null &
# $BLENDER --background --python $PYTHON_SCRIPT -- --subject 'rp_aneko_rigged_002' --split 'train' -o $SAVE_DIR -n 1 
