import numpy as np
import os
import glob
import tqdm
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sk_valid_names = [
    'hip', 'spine_01', 'spine_02', 'spine_03', 'neck', 'head', 'shoulder_l', 'shoulder_r',
    'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r', 'hand_l', 'hand_r', 'upperleg_l', 'upperleg_r',
    'lowerleg_l', 'lowerleg_r', 'foot_l', 'foot_r', 'lowerleg_twist_l', 'lowerleg_twist_r', 
    'upperleg_twist_l', 'upperleg_twist_r']
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--split', type=str, choices=['all', 'train', 'val'], default='all')
    parser.add_argument(
        '-m', '--method', type=str, choices=['kmeans'], default='kmeans')
    parser.add_argument(
        '-kl', '--klist', nargs="*", type=int, default=[50,])
    args = parser.parse_args()
    
    actions = np.loadtxt(f'../data/mixamo/{args.split}.txt', dtype=str)
    
    # load all
    sks, ids = [], []
    for action in tqdm.tqdm(actions):
        sk_files = glob.glob(f'../data/mixamo/skeletons/{action}/*.sk')
        for sk_file in sk_files:
            sk = np.loadtxt(sk_file, usecols=[1,2,3])
            sk_names = np.loadtxt(sk_file, usecols=[0], dtype=str).tolist()
            valid_rows = [sk_names.index(valid_name) for valid_name in sk_valid_names]
            hip = sk[sk_names.index('hip'), :]
            sk_aligned = sk[valid_rows, :] - hip
            
            h_min, h_max = sk_aligned[:, 1].min(), sk_aligned[:, 1].max()
            if (h_max - h_min) < 100:
                continue
            
            sks.append(sk_aligned.reshape(-1))
            ids.append([action, int(sk_file.split('/')[-1].replace('.sk', ''))])

    sks = np.stack(sks, axis=0)
    print (sks.shape, len(ids))
    
    # different visualize method
    sks_pca = PCA(n_components=2).fit_transform(sks)
    sks_tsne = TSNE(n_components=2, random_state=0).fit_transform(sks)
    
    # start kmeans
    for k in args.klist:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(sks)
        labels = kmeans.labels_
        
        # save
        save_file = f'../data/mixamo/clusters/{args.method}/{args.split}_{k}.json'
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        save_dict = {}
        for icluster in range(k):
            idxs = np.where(labels == icluster)[0]
            save_dict[icluster] = [ids[idx] for idx in idxs]

        print (f'save to {save_file}')
        with open(save_file, 'w') as f:
            json.dump(save_dict, f)
            
        # vis
        colors = cm.rainbow(np.linspace(0, 1, k))
        colors = [colors[l] for l in labels]
        plt.scatter(sks_tsne[:, 0], sks_tsne[:, 1], c=colors)
        plt.savefig(save_file.replace('.json', '.jpg'))