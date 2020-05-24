import sys
sys.path.append("../")

from lib.options import BaseOptions
args = BaseOptions().parse()

from lib.dataset import PIFuDataset
dataset = PIFuDataset(args, split='debug')
print (dataset.motion_list)
data_dict = dataset[0]

if args.num_sample_geo:
    dataset.visualize_sampling(data_dict, '../test_data/proj_geo.jpg', mode='geo')
if args.num_sample_color:
    dataset.visualize_sampling(data_dict, '../test_data/proj_color.jpg', mode='color')

dataset.visualize_sampling3D(data_dict, mode='color')
dataset.visualize_sampling3D(data_dict, mode='geo')

# speed 3.30 iter/s
# with tinyobj loader 5.27 iter/s
import tqdm
for _ in tqdm.tqdm(dataset):
    pass

