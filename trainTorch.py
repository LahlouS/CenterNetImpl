import sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import CardDlayingDataset, final_transforms, custom_collate
from model import CenterNet
from loss import focal_loss, get_predictions_from_head
from utils import print_base_img_annotated_img, plot_heatmaps, plot_loss_vs_epoch

OVERFIT_BATCH = False
arguments = sys.argv[1:]
if len(arguments) and arguments[0] == 'overfit':
    OVERFIT_BATCH = True


    
BASE_PATH = "/Users/sacha.lahlou/Library/CloudStorage/OneDrive-PMU/centernetImplementation/centernet/cardDetectionDataset/"
TEST_PATH = "test/"
TRAIN_PATH = "train/"
VAL_PATH = "valid/"
ANNOTATION_FILENAME = "_annotations.coco.json"

MODEL_SAVE_PATH = './model_parameters_save'


BATCH_SIZE = 2
INPUT_SHAPE = (3, 256, 256)
NBCLASSES = 53

size_loss_weight = 0.1
offset_loss_weight = 1
focal_loss_weight = 1

l1loss = torch.nn.L1Loss()

train_ds = CardDlayingDataset(BASE_PATH + TRAIN_PATH, transforms=final_transforms)
test_ds = CardDlayingDataset(BASE_PATH + TEST_PATH, transforms=final_transforms)

class_names = test_ds.classes_names
train_data_len = train_ds.__len__()

train_ds = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
test_ds = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)



EPCH = 10

if torch.backends.mps.is_available():
    print('running on device mps')
    device = torch.device('mps')
else:
    print('Warning: mps not available running on cpu')
    device = torch.device('cpu')

model = CenterNet(INPUT_SHAPE, NBCLASSES, need_fpn=False)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)#1e-3)


print('\n\nStart training... \n\n')
if OVERFIT_BATCH:
    over_batch = 0
    for batch in test_ds:
        over_batch = batch
        break
#     print('overfitting batch...')
#     for i in range(BATCH_SIZE):
#         print_base_img_annotated_img(over_batch['img'][i], 
#                                  over_batch['labels'][i]['boxes'],
#                                  over_batch['labels'][i]['labels'],
#                                  class_names
#                                  )


def train_one_step(batch):
    img =  batch['img'].to(device)
    # labels =  batch['labels'].to(device)
    hmaps = batch['hmaps'].to(device)
    off_maps = batch['offset_map'].to(device)
    size_maps = batch['size_map'].to(device)
    center_position_save = batch['center_position_save'].to(device)

    optimizer.zero_grad()

    hmaps_hat, off_maps_hat, size_map_hat = model(img)

    hmaps_loss = focal_loss(hmaps_hat, hmaps) # torchvision.ops.sigmoid_focal_loss(hmaps_hat, hmaps, reduction='mean') 

    size_loss = l1loss(get_predictions_from_head(size_map_hat, center_position_save), size_maps)
    off_loss = l1loss(get_predictions_from_head(off_maps_hat, center_position_save), off_maps)

    loss_det = hmaps_loss * focal_loss_weight + size_loss * size_loss_weight + off_loss * offset_loss_weight

    loss_det.backward()
    optimizer.step()
    return loss_det, hmaps_loss, size_loss, off_loss

loss_list = []
for epch_idx in range(EPCH):
    if OVERFIT_BATCH:
        FAKE_BATCH_SIZE = 120
        for batch_idx in range(FAKE_BATCH_SIZE):
            loss_det, hmaps_loss, size_loss, off_loss = train_one_step(over_batch)
            loss_list.append([loss_det.item(), hmaps_loss.item(), size_loss.item(), off_loss.item()])
            print(f'OVERFIT LOG: EPOCH: {epch_idx}/{EPCH} - BATCH: {batch_idx}/{FAKE_BATCH_SIZE} --> total loss: {loss_det} | focal_loss: {hmaps_loss}')
    else:
        for batch_idx, batch in enumerate(train_ds):
            loss_det, hmaps_loss, size_loss, off_loss = train_one_step(batch)
            loss_list.append([loss_det, hmaps_loss, size_loss, off_loss])
            print(f'LOG: EPOCH: {epch_idx}/{EPCH} - BATCH: {batch_idx}/{train_data_len // BATCH_SIZE} --> total loss: {loss_det} | focal_loss: {hmaps_loss}')

print('\n\nEnd training... \n\n')

loss_list = np.array(loss_list).reshape(len(loss_list[0]), -1)
plot_loss_vs_epoch(loss_list)

print('---> ', torch.save(model.state_dict(), MODEL_SAVE_PATH))
