import torch.nn as nn
import torch

from utils import intersection_over_union




class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, preds, target):
        preds = preds.reshape(-1, self.S, self.B, (self.C + 5 * self.B))

        # useful indexes:
        # 0-19: class probabilities
        # 20: class score
        # 21-25: four bounding-box values for first box
        # 26-29: four bounding-box values for second box
        iou_b1 = intersection_over_union(preds[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(preds[..., 26:30], target[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)

        exists_box = target[..., 20].unsqueeze(3) # Identity obj func (Iobj_i)

        # box coordinates
        box_preds = exists_box * (
            best_box * preds[..., 26:30] + (1 - best_box) * preds[..., 21:25]
        )

        box_targets = exists_box * target[..., 21:25]

        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(torch.abs(box_targets[..., 2:4]))

        # going from 3 dims to 1 for valid MSE input
        box_loss = self.mse(
            torch.flatten(box_preds, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # object loss
        pred_box = best_box * preds[..., 25:26] + (1 - best_box) * preds[..., 20:21]

        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # no object loss - take loss for both
        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * preds[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        no_obj_loss += (
            torch.flatten((1 - exists_box) * preds[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # class loss
        class_loss = self.mse(
            torch.flatten(exists_box * preds[..., 20], end_dim=-2),
            torch.flatten(exists_box * target[..., 20], end_dim=-2),
        )

        # actual loss
        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )
        return loss




