import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class residual(nn.Module):
    """
    residual block
    """
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x) # if downsampling, resize the feature map x
        return self.relu(bn2 + skip)


class Nonlocal(nn.Module):
    def __init__(self, dim, mapdim):
        super(Nonlocal, self).__init__()
        self.mapdim = dim
        self._init_layers(dim, mapdim)

    def _init_layers(self, dim, mapdim):
        self.conv1 = convolution(3, dim, dim) # b c h w 
        self.conv2   = convolution(3, dim, dim)
        self.phi = nn.Conv2d(dim, dim, 3, padding=1)
        self.theta = nn.Conv2d(dim, dim, 3, padding=1)
        self.g = nn.Conv2d(dim, dim, 3, padding=1)
        self.W = nn.Conv2d(dim, dim, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        # pdb.set_trace()

    def forward(self, x):
        x1 = self.conv1(x)
        # pdb.set_trace()
        b, c, h, w = x.size()
        g_x = self.g(x1).view(b, self.mapdim, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x1).view(b, self.mapdim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x1).view(b, self.mapdim, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.mapdim, h, w)

        z = self.W(y) + x
        z = self.conv2(z)
        return z

class Dilated(nn.Module):
    def __init__(self, dim, mapdim, dilate=3):
        super(Dilated, self).__init__()
        self.mapdim = mapdim
        self._init_layers(dim, mapdim, dilate)

    def _init_layers(self, dim, mapdim, dilate):
        self.p1_conv = convolution(3, dim, dim) # b c h w 
        self.p2_conv = nn.Conv2d(dim, dim, (3, 3), padding=(dilate, dilate), dilation=(dilate, dilate), bias=False)
        self.p2_bn   = nn.BatchNorm2d(dim)
        self.conv1   = nn.Conv2d(dim, dim, (1,1), bias=False)
        self.bn1     = nn.BatchNorm2d(dim)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv2 = convolution(3, dim, dim)

    def forward(self,x):
        conv = self.p1_conv(x)
        p2_conv = self.p2_conv(conv)
        p2_bn = self.p2_bn(p2_conv)
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1+p2_bn)
        return self.conv2(relu1)

class LandmarkP4(nn.Module):
    def __init__(self, dim, mapdim):
        super(LandmarkP4, self).__init__()
        # print('using pconv4 pooling checked ...')
        self.mapdim = mapdim
        self._init_layers(dim)

    def _init_layers(self, dim):
        # print(dim)
        # map to mapdim
        self.p1_conv1 = convolution(3, dim, self.mapdim)
        self.p2_conv1 = convolution(3, dim, self.mapdim)
        self.p3_conv1 = convolution(3, dim, self.mapdim)
        self.p4_conv1 = convolution(3, dim, self.mapdim)
        # map back to dim
        self.p_conv1 = nn.Conv2d(self.mapdim * 4, dim, (3,3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1   = nn.Conv2d(dim, dim, (1,1), bias=False)
        self.bn1     = nn.BatchNorm2d(dim)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv2   = convolution(3, dim, dim)
        from ._pconv.conv4 import TopLeftPool, TopRightPool, BottomLeftPool, BottomRightPool
        self.pool1 = TopRightPool()
        self.pool2 = TopLeftPool()
        self.pool3 = BottomLeftPool()
        self.pool4 = BottomRightPool()
    
    def forward(self, x):
        p1_conv1 = self.p1_conv1(x)
        p2_conv1 = self.p2_conv1(x)
        p3_conv1 = self.p3_conv1(x)
        p4_conv1 = self.p4_conv1(x)
        p1 = self.pool1(p1_conv1, torch.ones_like(p1_conv1))
        p2 = self.pool2(p2_conv1, torch.ones_like(p2_conv1))
        p3 = self.pool3(p3_conv1, torch.ones_like(p3_conv1))
        p4 = self.pool4(p4_conv1, torch.ones_like(p4_conv1))

        pool_feat = torch.cat([p1, p2, p3, p4], dim=1)
        p_conv1 = self.p_conv1(pool_feat)
        p_bn1   = self.p_bn1(p_conv1)
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)
        conv2 = self.conv2(relu1)
        return conv2

class LandmarkP1(nn.Module):
    def __init__(self, dim, mapdim):
        super(LandmarkP1, self).__init__()
        # print('using pconv4 pooling checked ...')
        self.mapdim = mapdim
        self._init_layers(dim)

    def _init_layers(self, dim):
        # map to mapdim
        self.p1_conv1 = convolution(3, dim, self.mapdim)
        # map back to dim
        self.p_conv1 = nn.Conv2d(self.mapdim, dim, (3,3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1   = nn.Conv2d(dim, dim, (1,1), bias=False)
        self.bn1     = nn.BatchNorm2d(dim)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv2   = convolution(3, dim, dim)
        from ._pconv.conv4 import TopLeftPool, TopRightPool, BottomLeftPool, BottomRightPool
        self.pool1 = TopRightPool()
        self.pool2 = TopLeftPool()
        self.pool3 = BottomLeftPool()
        self.pool4 = BottomRightPool()
    
    def forward(self, x):
        p1_conv1 = self.p1_conv1(x)
        p2_conv1 = self.p1_conv1(x)
        p3_conv1 = self.p1_conv1(x)
        p4_conv1 = self.p1_conv1(x)
        # bottom right region
        p1 = self.pool1(p1_conv1, torch.ones_like(p1_conv1))
        p2 = self.pool2(p2_conv1, torch.ones_like(p2_conv1))
        p3 = self.pool3(p3_conv1, torch.ones_like(p3_conv1))
        p4 = self.pool4(p4_conv1, torch.ones_like(p4_conv1))
        p1 = torch.max(p1, p2)
        p2 = torch.max(p3, p4)
        p1 = torch.max(p1, p2)
        pool_feat = p1
        p_conv1 = self.p_conv1(pool_feat)
        p_bn1   = self.p_bn1(p_conv1)
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)
        conv2 = self.conv2(relu1)
        return conv2


class LandmarkP2(nn.Module):
    def __init__(self, dim, mapdim):
        super(LandmarkP2, self).__init__()
        # print('using pconv4 pooling checked ...')
        self.mapdim = mapdim
        self._init_layers(dim)

    def _init_layers(self, dim):
        # map to mapdim
        self.p1_conv1 = convolution(3, dim, self.mapdim)
        self.p2_conv1 = convolution(3, dim, self.mapdim)
        # map back to dim
        self.p_conv1 = nn.Conv2d(self.mapdim * 2, dim, (3,3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1   = nn.Conv2d(dim, dim, (1,1), bias=False)
        self.bn1     = nn.BatchNorm2d(dim)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv2   = convolution(3, dim, dim)
        from ._pconv.conv4 import TopLeftPool, TopRightPool, BottomLeftPool, BottomRightPool
        self.pool1 = TopRightPool()
        self.pool2 = TopLeftPool()
        self.pool3 = BottomLeftPool()
        self.pool4 = BottomRightPool()
    
    def forward(self, x):
        p1_conv1 = self.p1_conv1(x)
        p2_conv1 = self.p1_conv1(x)
        p3_conv1 = self.p2_conv1(x)
        p4_conv1 = self.p2_conv1(x)
        # bottom right region
        p1 = self.pool1(p1_conv1, torch.ones_like(p1_conv1))
        p2 = self.pool2(p2_conv1, torch.ones_like(p2_conv1))
        p3 = self.pool3(p3_conv1, torch.ones_like(p3_conv1))
        p4 = self.pool4(p4_conv1, torch.ones_like(p4_conv1))
        p1 = torch.max(p1, p2)
        p2 = torch.max(p3, p4)

        pool_feat = torch.cat([p1, p2], dim=1)
        p_conv1 = self.p_conv1(pool_feat)
        p_bn1   = self.p_bn1(p_conv1)
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)
        conv2 = self.conv2(relu1)
        return conv2

class LandmarkP2x(nn.Module):
    def __init__(self, dim, mapdim):
        super(LandmarkP2x, self).__init__()
        # print('using pconv4 pooling checked ...')
        self.mapdim = mapdim
        self._init_layers(dim)

    def _init_layers(self, dim):
        # map to mapdim
        self.p1_conv1 = convolution(3, dim, self.mapdim)
        self.p2_conv1 = convolution(3, dim, self.mapdim)
        # map back to dim
        self.p_conv1 = nn.Conv2d(self.mapdim * 2, dim, (3,3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)
        self.conv1   = nn.Conv2d(dim, dim, (1,1), bias=False)
        self.bn1     = nn.BatchNorm2d(dim)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv2   = convolution(3, dim, dim)
        from ._pconv.conv4 import TopLeftPool, TopRightPool, BottomLeftPool, BottomRightPool
        self.pool1 = TopRightPool()
        self.pool2 = TopLeftPool()
        self.pool3 = BottomLeftPool()
        self.pool4 = BottomRightPool()
    
    def forward(self, x):
        p1_conv1 = self.p1_conv1(x)
        p2_conv1 = self.p2_conv1(x)
        p3_conv1 = self.p2_conv1(x)
        p4_conv1 = self.p1_conv1(x)
        # bottom right region
        p1 = self.pool1(p1_conv1, torch.ones_like(p1_conv1))
        p2 = self.pool2(p2_conv1, torch.ones_like(p2_conv1))
        p3 = self.pool3(p3_conv1, torch.ones_like(p3_conv1))
        p4 = self.pool4(p4_conv1, torch.ones_like(p4_conv1))
        p1 = torch.max(p1, p4)
        p2 = torch.max(p2, p3)
        pool_feat = torch.cat([p1, p2], dim=1)
        p_conv1 = self.p_conv1(pool_feat)
        p_bn1   = self.p_bn1(p_conv1)
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)
        conv2 = self.conv2(relu1)
        return conv2

class LandmarkP8(nn.Module):
    def __init__(self, dim, mapdim):
        super(Landmark8, self).__init__()
        # print('using pconv8 pooling checked ...')
        self.mapdim = mapdim
        self._init_layers(dim)

    def _init_layers(self, dim):
        # map to mapdim
        self.p1_conv1 = convolution(3, dim, self.mapdim)
        self.p2_conv1 = convolution(3, dim, self.mapdim)
        self.p3_conv1 = convolution(3, dim, self.mapdim)
        self.p4_conv1 = convolution(3, dim, self.mapdim)
        self.p5_conv1 = convolution(3, dim, self.mapdim)
        self.p6_conv1 = convolution(3, dim, self.mapdim)
        self.p7_conv1 = convolution(3, dim, self.mapdim)
        self.p8_conv1 = convolution(3, dim, self.mapdim)

        # map back to dim
        self.p_conv1 = nn.Conv2d(self.mapdim * 8, dim, (3,3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1   = nn.Conv2d(dim, dim, (1,1), bias=False)
        self.bn1     = nn.BatchNorm2d(dim)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv2   = convolution(3, dim, dim)
        
        from ._pconv.conv8 import I1Pool, I2Pool, I3Pool, I4Pool, I5Pool, I6Pool, I7Pool, I8Pool
        self.pool1 = I1Pool()
        self.pool2 = I2Pool()
        self.pool3 = I3Pool()
        self.pool4 = I4Pool()
        self.pool5 = I5Pool()
        self.pool6 = I6Pool()
        self.pool7 = I7Pool()
        self.pool8 = I8Pool()
    
    def forward(self, x, hook=None, hook_feat=None, hookdir=2):
        p1_conv1 = self.p1_conv1(x)
        p2_conv1 = self.p2_conv1(x)
        p3_conv1 = self.p3_conv1(x)
        p4_conv1 = self.p4_conv1(x)
        p5_conv1 = self.p5_conv1(x)
        p6_conv1 = self.p6_conv1(x)
        p7_conv1 = self.p7_conv1(x)
        p8_conv1 = self.p8_conv1(x)

        p1 = self.pool1(p1_conv1, torch.ones_like(p1_conv1))
        p2 = self.pool2(p2_conv1, torch.ones_like(p2_conv1))
        p3 = self.pool3(p3_conv1, torch.ones_like(p3_conv1))
        p4 = self.pool4(p4_conv1, torch.ones_like(p4_conv1))
        p5 = self.pool5(p5_conv1, torch.ones_like(p5_conv1))
        p6 = self.pool6(p6_conv1, torch.ones_like(p6_conv1))
        p7 = self.pool7(p7_conv1, torch.ones_like(p7_conv1))
        p8 = self.pool8(p8_conv1, torch.ones_like(p8_conv1))

        pool_feat = torch.cat([p1, p2, p3, p4, p5, p6, p7, p8], dim=1)
        p_conv1 = self.p_conv1(pool_feat)
        p_bn1   = self.p_bn1(p_conv1)
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)
        conv2 = self.conv2(relu1)
        return conv2


class LandmarkP4x(nn.Module):
    def __init__(self, dim, mapdim):
        super(LandmarkP4x, self).__init__()
        # print('using pconv4x pooling checked ...')
        self.mapdim = mapdim
        self._init_layers(dim)

    def _init_layers(self, dim):
        # map to mapdim
        self.p1_conv1 = convolution(3, dim, self.mapdim)
        self.p2_conv1 = convolution(3, dim, self.mapdim)
        self.p3_conv1 = convolution(3, dim, self.mapdim)
        self.p4_conv1 = convolution(3, dim, self.mapdim)

        # map back to dim
        self.p_conv1 = nn.Conv2d(self.mapdim * 4, dim, (3,3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1   = nn.Conv2d(dim, dim, (1,1), bias=False)
        self.bn1     = nn.BatchNorm2d(dim)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv2   = convolution(3, dim, dim)
        
        from ._pconv.conv8 import I1Pool, I2Pool, I3Pool, I4Pool, I5Pool, I6Pool, I7Pool, I8Pool
        self.pool1 = I1Pool()
        self.pool2 = I2Pool()
        self.pool3 = I3Pool()
        self.pool4 = I4Pool()
        self.pool5 = I5Pool()
        self.pool6 = I6Pool()
        self.pool7 = I7Pool()
        self.pool8 = I8Pool()
    
    def forward(self, x):
        p1_conv1 = self.p1_conv1(x)
        p2_conv1 = self.p2_conv1(x)
        p3_conv1 = self.p2_conv1(x)
        p4_conv1 = self.p3_conv1(x)
        p5_conv1 = self.p3_conv1(x)
        p6_conv1 = self.p4_conv1(x)
        p7_conv1 = self.p4_conv1(x)
        p8_conv1 = self.p1_conv1(x)

        p1 = self.pool1(p1_conv1, torch.ones_like(p1_conv1))
        p2 = self.pool2(p2_conv1, torch.ones_like(p2_conv1))
        p3 = self.pool3(p3_conv1, torch.ones_like(p3_conv1))
        p4 = self.pool4(p4_conv1, torch.ones_like(p4_conv1))
        p5 = self.pool5(p5_conv1, torch.ones_like(p5_conv1))
        p6 = self.pool6(p6_conv1, torch.ones_like(p6_conv1))
        p7 = self.pool7(p7_conv1, torch.ones_like(p7_conv1))
        p8 = self.pool8(p8_conv1, torch.ones_like(p8_conv1))
        p1 = torch.max(p1, p8)
        p2 = torch.max(p2, p3)
        p3 = torch.max(p4, p5)
        p4 = torch.max(p6, p7)
        pool_feat = torch.cat([p1, p2, p3, p4], dim=1)
        p_conv1 = self.p_conv1(pool_feat)
        p_bn1   = self.p_bn1(p_conv1)
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)
        conv2 = self.conv2(relu1)
        return conv2