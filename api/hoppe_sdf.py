from scipy.spatial import cKDTree

class HoppeSDF:
    def __init__(self, points, normals, faces=None):
        '''
        The HoppeSDF calculates signed distance towards a predefined oriented point cloud
        http://hhoppe.com/recon.pdf
        For clean and high-resolution pcl data, this is the fastest and accurate approximation of sdf
        :param points: pts
        :param normals: normals
        '''
        self.points = points
        self.faces = faces
        self.normals = normals
        self.kd_tree = cKDTree(self.points)
        self.len = len(self.points)
        
    def query(self, points):
        dists, idx = self.kd_tree.query(points)
        dirs = points - self.points[idx]
        signs = (dirs * self.normals[idx]).sum(axis=1)
        signs = (signs > 0) * 2 - 1
        return signs * dists

    def contains(self, points):
        sdf = self.query(points)
        labels = sdf < 0 # in is 1.0, out is 0.0
        return labels
