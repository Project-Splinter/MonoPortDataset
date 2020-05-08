# MonoPort Dataset
The folder structure is expected to be like this:

    # |- MonoPortDataset
    #     |- README.md
    #     |- init_link.sh # link mixamo and renderppl data to ./data
    #     |- bin/
    #     |- lib/
    #     |- scripts/
    #     |- data/
    #         |- mixamo/
    #             |- {all, train, val}.txt;
    #             |- actions  /<action>.fbx
    #             |- skeletons/<action>/%06d.sk
    #             |- clusters /kmeans/{all, train, val}_{k}.json
    #
    #         |- renderppl/
    #             |- {all, train, val}.txt;
    #             |- rigged    /<subject>_FBX/
    #             |- tpose_objs/<subject>.obj
    #             |- del_inside/<subject>/
    #                {del_faces.npy, del_verts.npy del_uv_mask.png}
    #
    #         |- pifu_orth/
    #             |- {train, val}.txt;
    #             |- <subject>/<action>/<frame>/
    #                {mesh.obj, skeleton.txt}
    #                {calib/*.txt, render/*.png, uv_render/*.jpg}
    #
    #     |- test_scripts/ # store experimental things.
    #     |- test_data/ # store experimental things.

# Dependence
- tqdm
- trimesh
- blender: 
```
cd ./MonoPortDataset
wget https://mirror.clarkson.edu/blender/release/Blender2.82/blender-2.82a-linux64.tar.xz -O ./bin/blender-2.82a-linux64.tar.xz
tar -xf ./bin/blender-2.82a-linux64.tar.xz -C ./bin/
```

# Note for scripts
`init_link.sh`: You need to setup the paths of where you store the data for renderppl and mixamo in this script. Then you can use this script to link the data into `./data/`.
```
# under ./MonoPortDataset/
sh init_link.sh;
```

`scripts/create_splits.sh`: This script is how we did the <train/val> split for both renderppl and mixamo. It will create `{all, train, val}.txt` for both renderppl and mixamo.
```
# under ./MonoPortDataset/scripts/
bash create_splits.sh;
```

`scripts/renderppl_tpose_objs.py`: We use this script to export renderppl data to obj from fbx file. We then use the obj files to find those verts/faces inside the mesh (mouth, teeth, eyebows etc.)
```
# under ./MonoPortDataset/scripts/
../bin/blender-2.82a-linux64/blender --background --python renderppl_tpose_objs.py > /dev/null
# or using this line for multi blender instances processing
bash ./blender_multi_instances.sh renderppl_tpose_objs.py 20
```

`scripts/renderppl_del_inside.py`: We use this script to find those verts/faces inside the mesh (mouth, teeth, eyebows etc.) By default it runs using 8 threads.
```
# under ./MonoPortDataset/scripts/
python renderppl_del_inside.py;
```

`scripts/mixamo_skeletons.py`: We use this script to export skeletons from mixamo data.
```
# under ./MonoPortDataset/scripts/
../bin/blender-2.82a-linux64/blender --background --python mixamo_skeletons.py > /dev/null
# or using this line for multi blender instances processing
bash ./blender_multi_instances.sh mixamo_skeletons.py 20
```

`scripts/mixamo_kmeans.py`: As mixamo action data has severely unbalanced distribution, we perform kmeans here on mixamo data to do the clustering first.
```
# under ./MonoPortDataset/scripts/
python mixamo_kmeans.py --split all --klist 10 20 50 100 200 300 500 1000
python mixamo_kmeans.py --split train --klist 10 20 50 100 200 300 500 1000
python mixamo_kmeans.py --split val --klist 10 20 50 100
```