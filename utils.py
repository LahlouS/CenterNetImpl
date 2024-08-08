
from functools import partial
from cjm_psl_utils.core import download_file, file_extract
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop
import io

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import torch.optim as optim
import matplotlib.pyplot as plt
torch.manual_seed(42)
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import cv2 as cv
# from google.colab.patches import cv2_imshow
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.autograd import Variable
from pytorch_lightning import loggers as pl_loggers



import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.v2  as transforms

## some tools
font_file = 'Oswald-Bold.ttf'
draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=15, colors="black")

classes_names_list = ['playing-cards', '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']


def print_base_img_annotated_img(img, box_xyxy, classes_id, classes_name_list, log_tensor=False):
    
    annotated_tensor = draw_bboxes(
        image = (img * 255).to(dtype=torch.uint8), 
        boxes = box_xyxy, 
        labels = [classes_name_list[int(c)] for c in classes_id],
    )

    img = np.transpose(np.array(img), (1, 2, 0))
    annotated_img = np.transpose(np.array(annotated_tensor), (1, 2, 0))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title(f'img {img.shape}')

    axs[1].imshow(annotated_img)
    axs[1].axis('off')
    axs[1].set_title(f'annotated {annotated_img.shape}')

    if not log_tensor:
        plt.tight_layout()
        plt.show()
    else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return fig, buf

def plot_heatmaps(matrices, titles=None, fig_title=None, figsize=(8, 6)):
    num_plots = len(matrices)
    fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0] * num_plots, figsize[1]))
    
    for i, matrix in enumerate(matrices):
        ax = axes[i] if num_plots > 1 else axes
        ax.imshow(matrix, cmap='viridis', interpolation='nearest')
        ax.set_title(titles[i] if titles else f"Heatmap {i+1}")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        # ax.colorbar()
    if fig_title:
        fig.suptitle(fig_title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.show()

def plot_loss_vs_epoch(loss_values_list, labels=['loss_det', 'hmaps_loss', 'size_loss', 'off_loss'], figsize=(8, 6)):
    epochs = np.array(range(1, len(loss_values_list[0]) + 1)) // 10

    num_plots = len(loss_values_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(figsize[0] * num_plots, figsize[1]))
    
    for i, loss_value in enumerate(loss_values_list):
        ax = axes[i] if num_plots > 1 else axes
        ax.plot(epochs, loss_value, 'b', label=labels[i])
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title(labels[i])

    plt.show()