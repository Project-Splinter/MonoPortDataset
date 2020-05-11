#!/bin/bash
NUM_THREADS=$1

echo "Runing multiple blender instances ->" $NUM_THREADS

PYTHON_SCRIPT='./pifu_orth_render.py'
BLENDER='../bin/blender-2.82a-linux64/blender'

cat ../data/renderppl/all.txt | shuf | xargs -P$NUM_THREADS -I {} $BLENDER --background --python $PYTHON_SCRIPT -- --subjects {} > /dev/null &
