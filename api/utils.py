import numpy as np

def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    return np.matmul(intrinsic, extrinsic)

