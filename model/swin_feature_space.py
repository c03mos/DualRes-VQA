from torch import nn
from torchvision import models
from torchvision.models import Swin_T_Weights


class Projection(nn.Module):
    def __init__(self,out_dim=256,in_dim=768):
        super(Projection, self).__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim,bias=False)
        self.linear2= nn.Linear(in_features=out_dim, out_features=out_dim,bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.norm(x)
        return  x

class FeatureExtractorSwin(nn.Module):
    def __init__(self,out_dim=256,in_dim=768):
        super(FeatureExtractorSwin, self).__init__()
        swin_t = models.swin_t(weights=Swin_T_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(swin_t.children())[:-1])
        self.projector = Projection(out_dim,in_dim)
    def forward(self,x):
        x_size = x.shape
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        x = self.feature_extractor(x)
        x = self.projector(x)
        B,C = x.shape
        x = x.view(x_size[0],x_size[1],C)
        return x

