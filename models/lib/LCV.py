import math
import torch
import torch.nn.functional as F
import numpy as np


class LearnableCorrBlock(torch.nn.Module):
    def __init__(self, dim, num_levels=4, radius=4):
        super(LearnableCorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.dim = dim

        self.raw_P = torch.nn.Parameter(torch.eye(self.dim), requires_grad=True)
        self.raw_D = torch.nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        self.register_buffer('eye', torch.eye(self.dim))

    def compute_cost_volume(self, fmap1, fmap2, maxdisp=4, warmup=False):
        if warmup:
            self.W = torch.eye(self.dim)
        else:
            # get matrix W
            self.raw_P_upper = torch.triu(self.raw_P)
            self.skew_P = (self.raw_P_upper - self.raw_P_upper.t()) / 2
            # Cayley representation, P is in Special Orthogonal Group SO(n)
            self.P = torch.matmul((self.eye + self.skew_P), torch.inverse(self.eye - self.skew_P))

            # obtain the diagonal matrix with positive elements
            self.trans_D = torch.atan(self.raw_D) * 2 / math.pi
            self.D = torch.diag((1 + self.trans_D) / (1 - self.trans_D))
            self.W = torch.matmul(torch.matmul(self.P.t(), self.D), self.P)

        b, c, h, w = fmap1.shape
        fmap1 = fmap1.view(b,c,h,w)[:,:,np.newaxis, np.newaxis]
        fmap2 = F.unfold(fmap2, (2*maxdisp+1,2*maxdisp+1), padding=maxdisp).view(b,c,2*maxdisp+1,2*maxdisp+1**2,h,w)
        # t = torch.tensordot(fmap1, self.W, dims=[[1],[0]])
        # print(t.shape)
        corr = torch.tensordot(fmap1, self.W, dims=[[1],[0]]).permute(0,5,1,2,3,4) * fmap2
        corr = corr.sum(1)
        b, ph, pw, h, w = corr.size()
        corr = corr.view(b, ph * pw, h, w)/fmap1.size(1)
        return corr

    # def forward(self, fmap1, fmap2, coords):
    def forward(self, fmap1, fmap2):
        return self.compute_cost_volume(fmap1, fmap2)

if __name__ == '__main__':
    def torch_pwc_corr(refimg_fea, targetimg_fea):
        maxdisp=4
        b,c,h,w = refimg_fea.shape
        targetimg_fea = F.unfold(targetimg_fea, (2*maxdisp+1,2*maxdisp+1), padding=maxdisp).view(b,c,2*maxdisp+1, 2*maxdisp+1**2,h,w)
        cost = refimg_fea.view(b,c,h,w)[:,:,np.newaxis, np.newaxis]*targetimg_fea.view(b,c,2*maxdisp+1, 2*maxdisp+1**2,h,w)
        cost = cost.sum(1)
        b, ph, pw, h, w = cost.size()
        cost = cost.view(b, ph * pw, h, w)/refimg_fea.size(1)
        return cost

    f1 = torch.randn(4, 32, 64, 208)
    f2 = torch.randn(4, 32, 64, 208)
    corr = LearnableCorrBlock(32)
    lcv = corr(f1, f2)
    print(lcv.shape)
    b, c ,h ,w = f1.shape
    maxdisp = 4
    cost = torch_pwc_corr(f1, f2)
    # fmap1 = f1.view(b,c,h,w)[:,:,np.newaxis, np.newaxis]
    # fmap2 = F.unfold(f2, (2*maxdisp+1,2*maxdisp+1), padding=maxdisp).view(b,c,2*maxdisp+1,2*maxdisp+1**2,h,w)
    # cost = fmap1 * fmap2 
    # cost = cost.sum(1)
    # b, ph, pw, h, w = cost.size()
    # cost = cost.view(b, ph * pw, h, w)/fmap1.size(1)
    print(torch.all(cost == lcv))

