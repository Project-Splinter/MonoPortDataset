import torch.nn as nn
import torch.nn.functional as F

class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False, momentum=0.9)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out

class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False, block4=True):
        super().__init__()
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        if block4:
            self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        else:
            self.block4 = None

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.last_op = nn.Sigmoid()

    def forward(self, p, z, c, **kwargs):
        # p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        if self.block4:
            net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        if self.last_op:
            out = self.last_op(out)

        return out

def CBN64b3(dim=3, c_dim=256):
    return DecoderCBatchNorm(
        dim=dim, z_dim=0, c_dim=c_dim, 
        hidden_size=64, leaky=False, legacy=False, block4=False)

def CBN128b3(dim=3, c_dim=256):
    return DecoderCBatchNorm(
        dim=dim, z_dim=0, c_dim=c_dim, 
        hidden_size=128, leaky=False, legacy=False, block4=False)

def CBN256b3(dim=3, c_dim=256):
    return DecoderCBatchNorm(
        dim=dim, z_dim=0, c_dim=c_dim, 
        hidden_size=256, leaky=False, legacy=False, block4=False)

def CBN64b4(dim=3, c_dim=256):
    return DecoderCBatchNorm(
        dim=dim, z_dim=0, c_dim=c_dim, 
        hidden_size=64, leaky=False, legacy=False, block4=True)

def CBN128b4(dim=3, c_dim=256):
    return DecoderCBatchNorm(
        dim=dim, z_dim=0, c_dim=c_dim, 
        hidden_size=128, leaky=False, legacy=False, block4=True)

def CBN256b4(dim=3, c_dim=256):
    return DecoderCBatchNorm(
        dim=dim, z_dim=0, c_dim=c_dim, 
        hidden_size=256, leaky=False, legacy=False, block4=True)

if __name__ == "__main__":
    import torch
    import tqdm

    model = CBN128b3(c_dim=256).cuda()

    p = torch.randn(16, 3, 4096).cuda()
    z = None
    c = torch.randn(16, 256, 4096).cuda()

    with torch.no_grad():
        # 61.37 (128) -> 95.37 (64) fps
        for _ in tqdm.tqdm(range(3000)):
            out = model(p, z, c)
            # print (out.shape)

