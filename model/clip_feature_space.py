import clip
from torch import nn


class Projection(nn.Module):
    def __init__(self,out_dim,in_dim):
        super(Projection, self).__init__()
        self.linear1=nn.Linear(in_features=in_dim, out_features=out_dim,bias=False)
        self.linear2=nn.Linear(in_features=out_dim, out_features=out_dim,bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.norm(x)
        return  x

class FeatureExtractorClip(nn.Module):
    def __init__(self,in_dim=512,out_dim=256):
        super(FeatureExtractorClip, self).__init__()
        model,_= clip.load("ViT-B/16")
        self.clip = model.visual
        self.projector = Projection(out_dim,in_dim)
    def forward(self,x):
        x_size = x.shape
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        x = self.clip(x)
        x = self.projector(x)
        x = x.view(x_size[0],x_size[1],-1)
        return x

