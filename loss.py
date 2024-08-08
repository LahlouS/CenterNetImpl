import torch


def focal_loss(pred_maps, gt_maps):

    positif_mask = gt_maps.eq(1).float()

    negatif_mask = gt_maps.lt(1).float()

    rev_prob_pow_beta = torch.pow(1 - gt_maps, 4)

    pred_maps = torch.clamp(pred_maps, 1e-12) # guaranty minimal value for numericall stability

    positive_instance_loss = ( torch.log(pred_maps) * torch.pow(1 - pred_maps, 2) ) * positif_mask # equivalent of if Y_hat_xyz == 1 then focalLoss
    negative_instance_loss = ( rev_prob_pow_beta * torch.pow(pred_maps, 2) * torch.log(1 - pred_maps) ) * negatif_mask # else

    npositif = positif_mask.sum() #  == N

    positive_instance_loss = positive_instance_loss.sum()
    negative_instance_loss = negative_instance_loss.sum()

    if npositif:
        return -(positive_instance_loss + negative_instance_loss) / npositif
    return -(negative_instance_loss)

def get_predictions_from_head(head, center_position_save):
    ret = torch.zeros(center_position_save.shape).to(head.device)

    for idx, (center_pos, pred) in enumerate(zip(center_position_save, head)):
        for i, ct in enumerate(center_pos):
            ret[idx][i] = pred[:, ct[0], ct[1]]
    return ret


