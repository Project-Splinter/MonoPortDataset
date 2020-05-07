#!/bin/bash
PYTHON_SCRIPT=$1
NUM_THREADS=$2
echo "Runing multiple blender instances ->" $NUM_THREADS

BLENDER='../bin/blender-2.82a-linux64/blender'

for ((i=1; i<=$NUM_THREADS; i++)); do
   $BLENDER --background --python $PYTHON_SCRIPT > /dev/null &
done