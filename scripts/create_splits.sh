#!/bin/bash

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# ====================================================
# renderppl 
RENDERPPL_DATA='../data/renderppl/rigged/'
RENDERPPL_ALL='../data/renderppl/all.txt'
RENDERPPL_VAL='../data/renderppl/val.txt'
RENDERPPL_TRAIN='../data/renderppl/train.txt'

# renderppl all list
ls $RENDERPPL_DATA > $RENDERPPL_ALL
sed -i 's/_FBX//g' $RENDERPPL_ALL

# renderppl val list
cat $RENDERPPL_ALL | xargs shuf -n20 -e --random-source=<(get_seeded_random 42) > $RENDERPPL_VAL

# renderppl train list
grep -Fvf $RENDERPPL_VAL $RENDERPPL_ALL > $RENDERPPL_TRAIN

# ====================================================
# mixamo 
MIXAMO_DATA='../data/mixamo/actions/'
MIXAMO_ALL='../data/mixamo/all.txt'
MIXAMO_VAL='../data/mixamo/val.txt'
MIXAMO_TRAIN='../data/mixamo/train.txt'

# mixamo all list
ls $MIXAMO_DATA > $MIXAMO_ALL
sed -i 's/.fbx//g' $MIXAMO_ALL

# mixamo val list
cat $MIXAMO_ALL | xargs shuf -n50 -e --random-source=<(get_seeded_random 42) > $MIXAMO_VAL

# mixamo train list
grep -Fvf $MIXAMO_VAL $MIXAMO_ALL > $MIXAMO_TRAIN