import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .ConvFilters import *
from .net_util import init_net
from .HRNetFilters import *
from .CBNClassifier import *
class ConvPIFuNet(BasePIFuNet):
    '''
    Conv Piximp network is the standard 3-phase network that we will use.
    The image filter is a pure multi-layer convolutional network,
    while during feature extraction phase all features in the pyramid at the projected location
    will be aggregated.
    It does the following:
        1. Compute image feature pyramids and store it in self.im_feat_list
        2. Calculate calibration and indexing on each of the feat, and append them together
        3. Classification.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(ConvPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'convpifu'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = self.define_imagefilter(opt)

        self.surface_classifier = self.define_classifier(opt)

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []

        init_net(self)

    def define_imagefilter(self, opt):
        net = None
        if opt.netIMF == 'multiconv':
            net = MultiConv(opt.enc_dim)
        elif 'resnet' in opt.netIMF:
            net = ResNet(model=opt.netIMF)
        elif opt.netIMF == 'vgg16':
            net = Vgg16()
        elif 'HRNet' in opt.netIMF:
            net = globals()[opt.netIMF]()
        else:
            raise NotImplementedError('model name [%s] is not recognized' % opt.netIMF)

        return net

    def define_classifier(self, opt):
        net = None
        if opt.classifierIMF == 'SurfaceClassifier':
            net = SurfaceClassifier(
                filter_channels=opt.mlp_dim,
                num_views=opt.num_views,
                no_residual=opt.no_residual,
                last_op=nn.Sigmoid())
        elif 'CBN' in opt.classifierIMF:
            net = globals()[opt.classifierIMF](opt.cly_dim, opt.filter_dim)
        else:
            raise NotImplementedError('model name [%s] is not recognized' % opt.classifierIMF)

        return net

    def filter(self, images, inplace=True):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        im_feat_list = self.image_filter(images)
        if inplace:
            self.im_feat_list = im_feat_list
        else:
            return im_feat_list

    def query(self, points, calibs, transforms=None, labels=None, im_feat_list=None, inplace=True):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]
        
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        z_feat = self.normalizer(z, calibs=calibs)

        # This is a list of [B, Feat_i, N] features
        if im_feat_list is None:
            im_feat_list = self.im_feat_list
        point_local_feat_list = [self.index(im_feat, xy) for im_feat in im_feat_list]
        
        if self.opt.classifierIMF == 'SurfaceClassifier':
            # [B, Feat_all, N]
            point_local_feat = torch.cat(point_local_feat_list + [z_feat], 1)
            preds = self.surface_classifier(point_local_feat)
        elif 'CBN' in self.opt.classifierIMF:
            point_local_feat = torch.cat(point_local_feat_list, 1)
            # xyz | z_feat (z / soft z)
            preds = self.surface_classifier(
                p=xyz if self.opt.cly_dim == 3 else z_feat, 
                z=None, c=point_local_feat) # [B, N]
            preds = preds.unsqueeze(1)
        # out of image plane is always set to 0
        preds = in_img[:,None].float() * preds
        if inplace:
            self.preds = preds
        else:
            return preds

    def forward(self, images, points, calibs, transforms=None, labels=None):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        # get the prediction
        res = self.get_preds()
        
        # get the error
        error = self.get_error()

        return res, error
