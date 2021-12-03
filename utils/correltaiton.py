import torch
import torch.nn as nn

# feat1 = torch.ones([8,196,7,16])
# feat2 = feat1.clone()

class MyCorrelation(nn.Module):
    def __init__(self, d=4):
        super(MyCorrelation,self).__init__()
        self.d = d
    
    def forward(self, feat1, feat2):
        h, w = feat1.size(2), feat1.size(3)
        cv = []
        feat2 = torch.nn.functional.pad(feat2, [self.d,self.d,self.d,self.d], "constant", 0)
        for i in range(2*self.d+1):
            for j in range(2*self.d+1):
                cv.append(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))
        
        return torch.cat(cv, axis=1)
      


def split_correlation(feat1, feat2, direction, d=4 ):
    assert direction == 'horizontal' or direction == 'vertical'
    h, w = feat1.size(2), feat1.size(3)
    cv = []
    if direction == 'horizontal':
        feat2 = torch.nn.functional.pad(feat2, [d,d,0,0], "constant", 0)
        for i in range(1):
            for j in range(2*d+1):
                cv.append(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))
    else:
        feat2 = torch.nn.functional.pad(feat2, [0,0,d,d], "constant", 0)
        for i in range(2*d+1):
            for j in range(1):
                cv.append(torch.mean(feat1*feat2[:,:,i:(i+h),j:(j+w)], dim=1, keepdim=True))

    cv = torch.cat(cv, axis=1)
    cv = torch.nn.functional.leaky_relu(input=cv, negative_slope=0.1, inplace=False)

    return cv

# since = time.time()
# cv = correlation(feat1.cuda(), feat2.cuda())
# print(time.time() - since)