import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from model.swin_feature_space import FeatureExtractorSwin
from model.clip_feature_space import FeatureExtractorClip



class GatedSpatioTempAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, sem, temp):
        combined = torch.cat([sem, temp], dim=-1)  # (B, T, 2H)

        Q = self.query(combined)
        K = self.key(combined)
        V = self.value(combined)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)

        attn_scores = F.softmax(attn_scores, dim=-1)  # (B, T, T)

        gate_input = torch.cat([sem, temp], dim=-1)

        gate = torch.sigmoid(self.gate_layer(gate_input))  # (B, T, 1)

        attn_weights = attn_scores * gate

        fused_feature = torch.bmm(attn_weights, V)

        return fused_feature + sem , attn_weights.mean(1)

class FusionCell(nn.Module):
    def __init__(self, ops_candidates, feat_dim):
        super().__init__()
        self.alpha = Parameter(torch.randn(len(ops_candidates)))
        self.ops = nn.ModuleList([
            self._build_op(op, feat_dim) for op in ops_candidates
        ])

    def _build_op(self, op_type, dim):
        if op_type == 'concat':
            return nn.Linear(2 * dim, dim)
        elif op_type == 'add':
            return nn.Linear(dim, dim)
        elif op_type == 'attn':
            return nn.MultiheadAttention(dim, 4)
        elif op_type == 'mul':
            return nn.LayerNorm(dim)
        elif op_type == 'modulate':
            return nn.Linear(dim, 2)
        else:
            raise ValueError(f"Invalid fusion op {op_type}!")

    def forward(self, x1, x2):
        if self.training:
            tau = 0.5
            weights = torch.softmax((self.alpha + torch.randn_like(self.alpha)) / tau, dim=-1)
        else:
            weights = torch.softmax(self.alpha, dim=-1)
        outputs = []
        for w, op in zip(weights, self.ops):
            if op.__class__.__name__ == 'Linear' and op.in_features == 2 * op.out_features:
                fused = op(torch.cat([x1, x2], dim=-1))
            elif isinstance(op, nn.MultiheadAttention):
                fused, _ = op(x1, x2, x2)
                fused = fused.squeeze(1)
            elif op.__class__.__name__ == 'Linear' and op.out_features == 2:
                feat = op(x2)
                alpha = torch.chunk(feat, 2, 2)[0]
                beta = torch.chunk(feat, 2, 2)[1]
                fused = torch.add(torch.mul(torch.abs(alpha), x1), beta)
            elif isinstance(op, nn.LayerNorm):
                fused = op(x1 * x2)
            else:
                fused = x1 + op(x2) if op.in_features == op.out_features else op(x1 + x2)
            outputs.append(w * fused)
        return sum(outputs)

class BaseModel(nn.Module):
    def __init__(self,ops_candidates):
        super().__init__()
        self.feature_extractor1 = FeatureExtractorClip()
        self.feature_extractor2 = FeatureExtractorSwin()
        self.AdaptiveGateFusionX = GatedSpatioTempAttention(256)
        self.AdaptiveGateFusionS = GatedSpatioTempAttention(256)
        self.fusion_cell = FusionCell(ops_candidates = ops_candidates,feat_dim=256)

        # self._load_stage1()
        self.score_mlp = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self,x,saliency,fast):
        # Feature Extractor
        x = self.feature_extractor1(x)

        ori_x = x

        saliency = self.feature_extractor2(saliency)

        ori_s = saliency

        # Spatial Fusion

        x,weight_x= self.AdaptiveGateFusionX(x,fast)

        saliency,weight_s= self.AdaptiveGateFusionS(saliency,fast)

        weight = 0.65 * weight_x + 0.35 * weight_s

        weight=F.softmax(weight,dim=-1)

        # Global Part Feature Fusion

        feat = self.fusion_cell(x, saliency)

        # Weight Score

        score = self.score_mlp(feat).squeeze(-1)

        score = weight * score

        return score.mean(1).squeeze(-1) ,ori_x, ori_s


if __name__ == '__main__':
    model = BaseModel()
    model.float()
    model.cuda()
    x = torch.randn(8,8,3,224,224).cuda()
    y = torch.randn(8, 8, 3, 224, 224).cuda()
    z = torch.randn(8, 8, 256).cuda()
    x,_,_ = model(x,y,z)
    print(x.shape)