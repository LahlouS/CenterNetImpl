# some tools:
import json
from functools import partial
from cjm_psl_utils.core import download_file, file_extract
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop
import math


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
torch.manual_seed(42)
import numpy as np
from PIL import Image
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.v2  as transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.autograd import Variable
from pytorch_lightning import loggers as pl_loggers


BASE_PATH = "/Users/sacha.lahlou/Library/CloudStorage/OneDrive-PMU/centernetImplementation/centernet/cardDetectionDataset/"
TEST_PATH = "test/"
TRAIN_PATH = "train/"
VAL_PATH = "valid/"
ANNOTATION_FILENAME = "_annotations.coco.json"

DS_MEAN = [0.5522, 0.5133, 0.4826]
DS_VAR = [0.0650, 0.0678, 0.0696]
NB_CLASS = 53

def custom_collate(batch):
    # print(type(batch), len(batch))
    ret = {
            'img': torch.cat([sample['img'].unsqueeze(0) for sample in batch]),

            'labels': [sample['labels'] for sample in batch],
            
            'hmaps': torch.cat([torch.tensor(sample['hmaps']).unsqueeze(0) for sample in batch]),

            'offset_map': torch.cat([torch.tensor(sample['offset_map']).unsqueeze(0) for sample in batch]),

            'size_map': torch.cat([torch.tensor(sample['size_map']).unsqueeze(0) for sample in batch]),
            
            'center_position_save': torch.cat([torch.tensor(sample['center_position_save']).unsqueeze(0) for sample in batch])
        }
    return ret


# DATA AUGMENTATIONS
TRAIN_SIZE = 256

data_aug_tfms = transforms.Compose(
    transforms=[
        CustomRandomIoUCrop(min_scale = 0.3, # minimum downscaling factor relatively to the input img; here 30%
                        max_scale = 1.0, # maximum factor
                        min_aspect_ratio = 0.5, # width cant be less than 50% of the height
                        max_aspect_ratio = 2.0, # cant be greater than 100%
                        sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], # controlling the probability of small IoU between box and ROI
                        trials = 400, # max trial for finding suitable crop
                        jitter_factor = 0.25 # randomness variance of the paramaters
        ),
        transforms.ColorJitter( # some color shuffle
                brightness = (0.875, 1.125),
                contrast = (0.5, 1.5),
                saturation = (0.5, 1.5),
                hue = (-0.05, 0.05),
        ),
        transforms.RandomGrayscale(), 
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5), # some rotations
    ],
)

# DATA TRANSFORMATIONS
resize_pad_tfm = transforms.Compose([
    ResizeMax(max_sz = TRAIN_SIZE), 
    PadSquare(shift=True),
    transforms.Resize([TRAIN_SIZE] * 2, antialias=True)
])

# DATA TRANSFORMATIONS
final_tfms = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=False), # will normalize the pixels
    # transforms.Normalize(mean=DS_MEAN, std=DS_VAR),
    transforms.SanitizeBoundingBoxes(),
])

# GATHERING EVERYTHING
final_transforms = transforms.Compose([
    # data_aug_tfms, 
    resize_pad_tfm, 
    final_tfms
])


# utils for centernet ETL

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1 = 1
  b1 = (height + width)
  c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  # r1 = (b1 + sq1) / 2 #
  r1 = (b1 - sq1) / (2 * a1)

  a2 = 4
  b2 = 2 * (height + width)
  c2 = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  # r2 = (b2 + sq2) / 2
  r2 = (b2 - sq2) / (2 * a2)

  a3 = 4 * min_overlap
  b3 = -2 * min_overlap * (height + width)
  c3 = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3 = (b3 + sq3) / 2
  # r3 = (b3 + sq3) / (2 * a3)

  # print('DEBUG gaussian radius:', 'r1:', r1, 'r2:', r2, 'r3', r3)
  return min(r1, r2, r3)

# DEBUG gaussian radius: r1: 0.49064965930913207 r2: 0.4309534842669742 r3 1.4711701434024533
# DEBUG gaussian radius: r1: 0.5677848597182322 r2: 0.5                 r3 1.689346597454362
# DEBUG gaussian radius: r1: 0.5976076634878256 r2: 0.5274093219876037 r3 1.766922287382016
# DEBUG gaussian radius: r1: 0.5550887221618188 r2: 0.4900199203977733 r3 1.6399203184089064

def gaussian2D(shape, sigma=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]

  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

class CardDlayingDataset(Dataset):
    def __init__(self, path, transforms=None):
        super().__init__()
        self.path = path

        with open(self.path + '_annotations.coco.json', 'r') as f:
            datas = json.load(f)
        classes = datas['categories']
        imgs = datas['images']
        annotations = datas['annotations']
        
        self.parsed_datas = self._datas_parser(imgs, annotations)
        self.classes_names = [c['name'] for c in classes] # class_name are already in order

        
        self.size = len(self.parsed_datas)
        self.transforms = transforms

        self.max_objs = 5
        self.padding = 127  # 31 for resnet/resdcn
        self.downsampling_ratio = 4
        self.img_size = {'h': TRAIN_SIZE, 'w': TRAIN_SIZE}
        self.fmap_size = {'h': TRAIN_SIZE // self.downsampling_ratio, 'w': TRAIN_SIZE // self.downsampling_ratio}
        # self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.15


        
    def __getitem__(self, i):
        img = Image.open(self.path + self.parsed_datas[i]['filename']).convert('RGB')
        labels = self.parsed_datas[i]['labels']
        
        
        if self.transforms:
            bx = [b['box'] for b in labels] # [ [x,y,w,h], [x,y,w,h], ...]
            cid = [c['classe'] for c in labels] # [id, id, id, ..]

            # make it 'consumable' for our transformations 
            targets_data_struct = {
                'boxes': BoundingBoxes(torchvision.ops.box_convert(torch.Tensor(bx), 'xywh', 'xyxy'),
                           format='xyxy',
                           canvas_size=img.size
                           ),
                'labels': torch.Tensor(cid)
            }

            hmaps = np.zeros( (len(self.classes_names), self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32 )
            offset_map = np.zeros( (self.max_objs, 2), dtype=np.float32 )
            size_map = np.zeros( (self.max_objs, 2), dtype=np.float32 )
            center_position_save = np.zeros( (self.max_objs, 2), dtype=np.int32 )

            img, labels = self.transforms(img, targets_data_struct)

            for i, (bbox, label) in enumerate(zip(labels['boxes'], labels['labels'])):
                p = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]) # (x, y) coordinates of the center of the boundingbox
                p_down = (p / self.downsampling_ratio)
                p_down_int = p_down.astype(np.int32) # the downsampled centerpoint p_tilde = (p / R) from the paper
                h_down, w_down = (bbox[3] - bbox[1]) / self.downsampling_ratio, (bbox[2] - bbox[0]) / self.downsampling_ratio # height and width of the downsampled bounding box

                std = max(0, int(gaussian_radius((math.ceil(h_down), math.ceil(w_down)), self.gaussian_iou )))
                
                draw_umich_gaussian(hmaps[int(label)], p_down_int, std)

                size_map[i] = torch.tensor([w_down, h_down], dtype=torch.float32)
                offset_map[i] = torch.tensor(p_down - p_down_int)
                center_position_save[i] = torch.tensor(p_down_int)
            
            return {
                'img': img,
                'labels': labels,
                'hmaps': hmaps,
                'offset_map': offset_map,
                'size_map': size_map,
                'center_position_save' : center_position_save
            }
        
        return (img, labels)

    def __len__(self):
        return self.size
    
    def _datas_parser(self, imgs, annotations):
        images_list_parsed = []

        for img in imgs:
            img_id, filename = img['id'], img['file_name']
            related_annotations = []
            for annot in annotations:
                if annot["image_id"] == img_id:
                    related_annotations.append(dict( box=annot['bbox'], classe=annot['category_id'], area=annot['area']) )
            images_list_parsed.append(dict(id=img_id, filename=filename, labels=related_annotations))

        return (images_list_parsed)
    

class CardDlayingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.BASE_PATH = "/Users/sacha.lahlou/Library/CloudStorage/OneDrive-PMU/centernetImplementation/centernet/cardDetectionDataset/"
        self.TEST_PATH = "test/"
        self.TRAIN_PATH = "train/"
        self.VAL_PATH = "valid/"
        self.ANNOTATION_FILENAME = "_annotations.coco.json"

        self.batch_size = batch_size

        # maybe we will add transformations in a config file and pass it as arguments of the module

        # DATA AUGMENTATIONS
        data_aug_tfms = transforms.Compose(
            transforms=[
                CustomRandomIoUCrop(min_scale = 0.3, # minimum downscaling factor relatively to the input img; here 30%
                              max_scale = 1.0, # maximum factor
                              min_aspect_ratio = 0.5, # width cant be less than 50% of the height
                              max_aspect_ratio = 2.0, # cant be greater than 100%
                              sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], # controlling the probability of small IoU between box and ROI
                              trials = 400, # max trial for finding suitable crop
                              jitter_factor = 0.25 # randomness variance of the paramaters
                ),
                transforms.ColorJitter( # some color shuffle
                        brightness = (0.875, 1.125),
                        contrast = (0.5, 1.5),
                        saturation = (0.5, 1.5),
                        hue = (-0.05, 0.05),
                ),
                transforms.RandomGrayscale(), 
                transforms.RandomEqualize(),
                transforms.RandomPosterize(bits=3, p=0.5),
                transforms.RandomHorizontalFlip(p=0.5), # some rotations
            ],
        )

        # DATA TRANSFORMATIONS
        resize_pad_tfm = transforms.Compose([
            ResizeMax(max_sz = TRAIN_SIZE), 
            PadSquare(shift=True),
            transforms.Resize([TRAIN_SIZE] * 2, antialias=True)
        ])

        # DATA TRANSFORMATIONS
        final_tfms = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True), # will normalize the pixels
            # transforms.Normalize(mean=DS_MEAN, std=DS_VAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.SanitizeBoundingBoxes(),
        ])

        # GATHERING EVERYTHING
        self.final_transforms = transforms.Compose([
            # data_aug_tfms, 
            resize_pad_tfm, 
            final_tfms
        ])

    def prepare_data(self):
        CardDlayingDataset(self.BASE_PATH + self.TRAIN_PATH, self.final_transforms) # do we need to differentiate transformation between train / test / valid datasets ?
        CardDlayingDataset(self.BASE_PATH + self.TEST_PATH, self.final_transforms)
        CardDlayingDataset(self.BASE_PATH + self.VAL_PATH, self.final_transforms)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.card_training_ds = CardDlayingDataset(self.BASE_PATH + self.TRAIN_PATH, self.final_transforms)
            self.card_validation_ds = CardDlayingDataset(self.BASE_PATH + self.VAL_PATH, self.final_transforms) # I think no need to split our DS since we have a repo for the validation data
        elif stage == 'test':
            self.card_test_ds = CardDlayingDataset(self.BASE_PATH + self.TEST_PATH, self.final_transforms)

    def train_dataloader(self):
        return DataLoader(self.card_training_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._custom_collate)

    def val_dataloader(self):
        return DataLoader(self.card_training_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._custom_collate)

    def test_dataloader(self):
        return DataLoader(self.card_training_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._custom_collate)

    def predict_dataloader(self):
        return DataLoader(self.card_training_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._custom_collate)
    
    def _custom_collate(self, batch):
        # print(type(batch), len(batch))
        ret = {
                'img': torch.cat([sample['img'].unsqueeze(0) for sample in batch]),

                'labels': [sample['labels'] for sample in batch],
                
                'hmaps': torch.cat([torch.tensor(sample['hmaps']).unsqueeze(0) for sample in batch]),

                'offset_map': torch.cat([torch.tensor(sample['offset_map']).unsqueeze(0) for sample in batch]),

                'size_map': torch.cat([torch.tensor(sample['size_map']).unsqueeze(0) for sample in batch]),
                
                'center_position_save': torch.cat([torch.tensor(sample['center_position_save']).unsqueeze(0) for sample in batch])
            }
        return ret
    
    
if __name__ == '__main__':
    dm = CardDlayingDataModule(batch_size=32)
    print('TEST: instanciation happen well')
    dm.setup('test')
    test = dm.test_dataloader()
    for batch in test:
        hmap, offmap, sizemap, center_position_save = batch['hmaps'], batch['offset_map'], batch['size_map'], batch['center_position_save']
        print('hmap.shape --> ', hmap.shape)
        print('offmap.shape --> ', offmap.shape)
        print('sizemap.shape --> ', sizemap.shape)
        print('center_position_save.shape --> ', center_position_save.shape)
        break
    print('TEST: DataLoader load well')
