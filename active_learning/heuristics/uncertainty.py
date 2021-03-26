# @Author: Pieter Blok
# @Date:   2021-03-25 15:06:20
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-03-25 15:40:13

import numpy as np
import torch

def uncertainty(observations, iterations, width, height, device):
    uncertainty_list = []
    
    for key, val in observations.items():
        softmaxes = [v['softmaxes'] for v in val]
        entropies = torch.stack([torch.distributions.Categorical(softmax).entropy() for softmax in softmaxes])

        mean_bbox = torch.mean(torch.stack([v['pred_boxes'].tensor for v in val]), axis=0)
        mean_mask = torch.mean(torch.stack([v['pred_masks'].flatten().type(torch.cuda.FloatTensor) for v in val]), axis=0)
        mean_mask[mean_mask <= 0.3] = 0.0
        mean_mask = mean_mask.reshape(-1, width, height)

        mask_IOUs = []
        for v in val:
            current_mask = v['pred_masks']
            overlap = torch.logical_and(mean_mask, current_mask)
            union = torch.logical_or(mean_mask, current_mask)
            if union.sum() > 0:
                IOU = torch.divide(overlap.sum(), union.sum())
                mask_IOUs.append(IOU.unsqueeze(0))

        mask_IOUs = torch.cat(mask_IOUs)

        bbox_IOUs = []
        mean_bbox = mean_bbox.squeeze(0)
        boxAArea = torch.multiply((mean_bbox[2] - mean_bbox[0] + 1), (mean_bbox[3] - mean_bbox[1] + 1))
        for v in val:
            current_bbox = v['pred_boxes'].tensor.squeeze(0)
            xA = torch.max(mean_bbox[0], current_bbox[0])
            yA = torch.max(mean_bbox[1], current_bbox[1])
            xB = torch.min(mean_bbox[2], current_bbox[2])
            yB = torch.min(mean_bbox[3], current_bbox[3])
            interArea = torch.multiply(torch.max(torch.tensor(0).to(device), xB - xA + 1), torch.max(torch.tensor(0).to(device), yB - yA + 1))
            boxBArea = torch.multiply((current_bbox[2] - current_bbox[0] + 1), (current_bbox[3] - current_bbox[1] + 1))
            bbox_IOU = torch.divide(interArea, (boxAArea + boxBArea - interArea))
            bbox_IOUs.append(bbox_IOU.unsqueeze(0))

        bbox_IOUs = torch.cat(bbox_IOUs)

        val_len = torch.tensor(len(val)).to(device)
        outputs_len = torch.tensor(iterations).to(device)

        # u_sem = torch.clamp(1-torch.max(entropies), min=0, max=1)
        u_sem = torch.clamp(1-torch.mean(entropies), min=0, max=1)
        u_spl_m = torch.clamp(torch.divide(mask_IOUs.sum(), val_len), min=0, max=1)
        u_spl_b = torch.clamp(torch.divide(bbox_IOUs.sum(), val_len), min=0, max=1)
        
        try:
            u_n = torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1)
        except:
            u_n = 0.0

        u_h_m = torch.multiply(u_sem, u_spl_m)
        u_b_w = torch.multiply(u_spl_b, u_n)
        u_h_m_w = torch.multiply(u_h_m, u_n)
        u_h_m_b_w = torch.multiply(u_h_m, u_b_w)
        
        uncertainty_list.append(u_h_m_b_w.unsqueeze(0))

    if uncertainty_list:
        uncertainty_list = torch.cat(uncertainty_list)
        uncertainty = torch.mean(uncertainty_list)
        # uncertainty = torch.min(uncertainty_list)
    else:
        uncertainty = torch.tensor([float('NaN')]).to(device)

    return uncertainty.detach().cpu().numpy().squeeze(0)