import torch
import torch.nn.functional as F
from torch import nn


class DecoupledCircleLoss(nn.Module):
    def __init__(self, gamma_p=64, gamma_n=64, margin_p=0.1, margin_n=0.5, lambda_weight=0.5):
        super().__init__()
        self.gamma_p = gamma_p
        self.gamma_n = gamma_n
        self.margin_p = margin_p
        self.margin_n = margin_n
        self.lambda_weight = lambda_weight

    def forward(self, sp, sn):
        alpha_p = F.relu(sp.detach() - self.margin_p)
        alpha_n = F.relu(self.margin_n - sn.detach())

        logit_p = -self.gamma_p * alpha_p * (sp - self.margin_p)
        loss_intra = torch.log(1 + torch.sum(torch.exp(logit_p), dim=-1))

        logit_n = self.gamma_n * alpha_n * (sn - self.margin_n)
        loss_inter = torch.log(1 + torch.sum(torch.exp(logit_n), dim=-1))

        return self.lambda_weight * loss_intra.mean() + (1 - self.lambda_weight) * loss_inter.mean()


class ProjectorLoss(nn.Module):
    def __init__(self,margin_p=.35,margin_n=1.5):
        super().__init__()
        self.margin_p = margin_p
        self.margin_n = margin_n
        self.criterion = DecoupledCircleLoss()

    def forward(self, features, mos):
        B, f, C = features.shape
        features = features / torch.norm(features, p=2, dim=2, keepdim=True).clamp(min=1e-8)  # (batch, frame, feature)
        features = features.permute(1, 0, 2)
        sim_per_frame = torch.bmm(features, features.transpose(1, 2))
        all_similar = sim_per_frame.mean(dim=0)  # (batch, batch)
        mos_diff = torch.abs(mos.unsqueeze(1) - mos.unsqueeze(0))
        eye_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        positive_mask = (mos_diff < 0.35) & ~eye_mask
        negative_mask = (mos_diff > 1.5) & ~eye_mask
        mos_diff_pos = mos_diff.clone()
        mos_diff_pos[eye_mask] = float('inf')
        pos_min_values, pos_min_indices = torch.min(mos_diff_pos, dim=1)
        no_pos = positive_mask.sum(dim=1) == 0
        positive_mask[no_pos, pos_min_indices[no_pos]] = True
        temp_mos_diff_neg = mos_diff.clone()
        temp_mos_diff_neg[positive_mask | eye_mask] = float('-inf')
        neg_max_values, neg_max_indices = torch.max(temp_mos_diff_neg, dim=1)
        no_neg = negative_mask.sum(dim=1) == 0
        negative_mask[no_neg, neg_max_indices[no_neg]] = True
        conflict_mask = positive_mask & negative_mask
        negative_mask &= ~conflict_mask
        pos_sim_matrix = all_similar * positive_mask.float()
        neg_sim_matrix = all_similar * negative_mask.float()
        return self.criterion(pos_sim_matrix, neg_sim_matrix)






