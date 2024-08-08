import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import io
from loguru import logger

classes_names = ['playing-cards', '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def topKscoresPerBatch(fmaps, K=40):
    batch, channels, height, width = fmaps.shape

    # first we want the top K per fmaps
    flattened_hmaps = fmaps.reshape(batch, channels, -1)
    topk_scores_per_cls, topk_indices = torch.topk(flattened_hmaps, K)

    # computing x and y in (h, w) space
    topk_indices = topk_indices % (height * width)
    topk_y = (topk_indices / width).int().float()  
    topk_x = (topk_indices % width).int().float()

    # now we want the topk all classes merged, for each batch separatly
    flattened_cls = topk_scores_per_cls.reshape(batch, -1)
    topk_scores, indices = torch.topk(flattened_cls, K)

    topk_cls = (indices / K).int() # compute wich cls the topk belong

    # updating indices, x and y by matching indices and previous top40 for each hmaps

    topk_indices = gather_feature(topk_indices.view(batch, -1, 1), indices).reshape(batch, K)
    topk_x = gather_feature(topk_x.view(batch, -1, 1), indices).reshape(batch, K)
    topk_y = gather_feature(topk_y.view(batch, -1, 1), indices).reshape(batch, K)

    return topk_scores, topk_indices, topk_cls, topk_y, topk_x

def pool_nms(hm, pool_size=3):
    pad = (pool_size - 1) // 2
    hm_max = F.max_pool2d(hm, pool_size, stride=1, padding=pad)
    keep = (hm_max == hm).float()
    return hm * keep

def generate_heatmap_to_buf(heatmaps, title=None):

    # Convert tensor to numpy array
    heatmaps = heatmaps.numpy()

    # Create a single subplot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the heatmap
    ax.imshow(heatmaps, cmap='hot', interpolation='nearest')
    ax.set_title(title if title else "no title")
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf, fig



def inference(hmaps_hat, off_maps_hat, size_map_hat, input_img_h, input_img_w, topk=5, activation_treshold=0, debug=False):
            
            img_h, img_w = input_img_h, input_img_w
            batch_size, nchannels, out_h, out_w = hmaps_hat.shape

            pooled_hmaps_hat = pool_nms(hmaps_hat, 3) # NMS
            topk_scores, indices, clas, y, x = topKscoresPerBatch(pooled_hmaps_hat, topk) # shape == (nb_batch, topk)
            
            offsets = gather_feature(off_maps_hat, indices, use_transform=True) # match indices (batch_size, K) with indices of the topK spotted peaks
            offsets = offsets.reshape(batch_size, topk, 2) # should be useless

            # update our (x, y) pos with the predicted offset
            xs = x.view(batch_size, topk, 1) + offsets[:, :, 0].view(batch_size, topk, 1)
            ys = y.view(batch_size, topk, 1) + offsets[:, :, 1].view(batch_size, topk, 1)

            size_reg = gather_feature(size_map_hat, indices, use_transform=True)
            size_reg = size_reg.reshape(batch_size, topk, 2) # should be useless

            clas = clas.reshape(batch_size, topk, 1).float()
            topk_scores = topk_scores.reshape(batch_size, topk, 1)

            half_w , half_h = size_reg[:, :, 0:1] / 2, size_reg[:, :, 1:2] / 2
            half_w , half_h = half_w.reshape(batch_size, topk, 1) , half_h.reshape(batch_size, topk, 1)

            pred_bboxes_xyxy = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim = 2) # (batch_size, topk, [x_inf, y_inf, x_sup, y_sup])

            detect = []
            for batch_idx in range(batch_size):
                mask = topk_scores[batch_idx].gt(activation_treshold) # create a binary mask of minimum accepted activations

                batch_boxes = pred_bboxes_xyxy[batch_idx][mask.squeeze(-1), :]
                # upsampling bbox to input img size
                batch_boxes[:, [0, 2]] *= img_w / out_w
                batch_boxes[:, [1, 3]] *= img_h / out_h

                batch_scores = topk_scores[batch_idx][mask]

                batch_clas = clas[batch_idx][mask]
                # batch_clas = [classes_names[int(cid.item())] for cid in batch_clas]
                detect.append([batch_boxes, batch_scores, batch_clas, pooled_hmaps_hat[batch_idx]])
            
            if not debug:
                return detect
            else:
                return {
                    'detect': detect,
                    'topKsize': size_reg,
                    'topKoffset': offsets,
                    'topKx': xs,
                    'topKy': ys
                }