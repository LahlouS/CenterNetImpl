import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import MultiStepLR

from PIL import Image

from dataset import CardDlayingDataModule
from model import CenterNet
from loss import focal_loss, get_predictions_from_head
from inferance import generate_heatmap_to_buf, inference #, plt_to_tensor
from utils import print_base_img_annotated_img, classes_names_list
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.autograd import Variable
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.detection import MeanAveragePrecision
from loguru import logger


class CenterNetpl(pl.LightningModule):
    def __init__(self, batch_size, 
                        input_shape = (3, 256, 256),
                        nclasses = 53,
                        size_loss_weight = 0.1,
                        offset_loss_weight = 1,
                        focal_loss_weight = 1,
                        overfit_mode=False,
                        topk=4,
                        activation_treshold = 0.0
                        ):
        
        super().__init__()
        self.batch_size = batch_size


        self.nclasses = nclasses
        self.input_shape = input_shape
        self.model = CenterNet(input_shape=input_shape, nclasses = self.nclasses, need_fpn=True)

        self.focal_loss = focal_loss
        self.l1loss = torch.nn.L1Loss()

        self.size_loss_weight = size_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.focal_loss_weight = focal_loss_weight

        self.overfit_mode = overfit_mode

        self.topk = topk
        self.activation_treshold = activation_treshold

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.2)
        return [optimizer], [scheduler]
    
    def forward(self, x):
        hmaps, off_maps, size_map = self.model(x)
        return hmaps, off_maps, size_map
    
    def _step(self, batch):
        img, labels, hmaps, off_maps, size_maps, center_position_save = batch['img'], batch['labels'], batch['hmaps'], batch['offset_map'], batch['size_map'], batch['center_position_save']

        hmaps_hat, off_maps_hat, size_map_hat = self(img)

        hmaps_loss = self.focal_loss(hmaps_hat, hmaps)
        # torchvision.ops.sigmoid_focal_loss(hmaps_hat, hmaps, reduction='mean') 
        # print(size_map_hat.device, center_position_save.device, size_maps.device)

        size_loss = self.l1loss(get_predictions_from_head(size_map_hat, center_position_save), size_maps)
        off_loss = self.l1loss(get_predictions_from_head(off_maps_hat, center_position_save), off_maps)

        loss_det = hmaps_loss * self.focal_loss_weight + size_loss * self.size_loss_weight + off_loss * self.offset_loss_weight

        return {
                'loss_det': loss_det, 
                'hmaps_loss': hmaps_loss, 
                'size_loss': size_loss, 
                'off_loss': off_loss,
                'hmaps_hat': hmaps_hat,
                'off_maps_hat': off_maps_hat,
                'size_map_hat': size_map_hat,
                'batch': batch
                }
    ################# TRAINING ############################################
    def training_step(self, train_batch, batch_idx):
        ret = self._step(train_batch)
        
        # add some specific lof for training
        self.log('train/hmaps_loss', ret['hmaps_loss'])
        self.log('train/size_loss', ret['size_loss'])
        self.log('train/off_loss', ret['off_loss'])
        self.log('train/loss_det', ret['loss_det'], on_step=True, prog_bar=True, batch_size=self.batch_size)
        
        # inferate on last predictions
        self.last_preds_batch = ret['hmaps_hat'], ret['off_maps_hat'], ret['size_map_hat'], ret['batch']

        # save the overfitting parameters if overfitting mode
        if self.overfit_mode :
            self.debug_batch = train_batch

        return ret['loss_det']
    
    # will probably need to remove this one as it will be called every n batch, it work for the moment because I'm overfitting over only one batch
    def on_train_batch_start(self, batch, batch_idx):
        if not self.trainer.current_epoch:
            imgs = batch['img'].detach().cpu()
            labels = batch['labels']
            hmaps = batch['hmaps'].detach().cpu()
            # off_maps = batch['offset_map']
            # size_maps = batch['size_map']
            # center_position_save = batch['center_position_save']

            # PRINTING IMAGE AND ANOTATION
            self._log_image_to_tensorboard(imgs, labels, need_detach_gpu=True, title='train/GT:')

    def on_train_end(self):
        print('training is over')
        hmaps_hat, off_maps_hat, size_map_hat, batch = self.last_preds_batch
        
        hmaps_hat = hmaps_hat.detach().cpu()
        off_maps_hat = off_maps_hat.detach().cpu()
        size_map_hat = size_map_hat.detach().cpu()

        # img = batch['img'].cpu()
        img = batch['img']

        img_h, img_w = img.shape[2], img.shape[3] # img shape is (batch, 3, 256, 256)
        detect = inference(hmaps_hat, off_maps_hat, size_map_hat, img_h, img_w, topk=self.topk, activation_treshold=self.activation_treshold, debug=False)

        # to fit Dataset way of formating data
        pred_labels = self._format_det_output(detect)
        self._log_image_to_tensorboard(img, pred_labels, need_detach_gpu=True, title='train/PREDICTION:') # logging prediction
        self._log_hmaps_to_tensorboard(pred_labels, title='train/') # logging topk activated hmaps
        mAP = self._compute_map(pred_labels, batch, debug=True)

    ################# ######## ############################################


    def test_step(self, train_batch, batch_idx):
        ret = self._step(train_batch)
        
        # add some specific lof for training
        self.log('test/hmaps_loss', ret['hmaps_loss'])
        self.log('test/size_loss', ret['size_loss'])
        self.log('test/off_loss', ret['off_loss'])
        self.log('test/loss_det', ret['loss_det'], on_step=True, prog_bar=True)

        return ret['loss_det']

    def validation_step(self, train_batch, batch_idx):
        ret = self._step(train_batch)

        hmaps_hat = ret['hmaps_hat']
        off_maps_hat = ret['off_maps_hat']
        size_map_hat = ret['size_map_hat']
        batch = ret['batch']
        
        hmaps_hat = hmaps_hat.detach().cpu()
        off_maps_hat = off_maps_hat.detach().cpu()
        size_map_hat = size_map_hat.detach().cpu()

        img = batch['img']

        img_h, img_w = img.shape[2], img.shape[3] # img shape is (batch, 3, 256, 256)
        detect = inference(hmaps_hat, off_maps_hat, size_map_hat, img_h, img_w, topk=self.topk, activation_treshold=self.activation_treshold, debug=False)

        pred_labels = self._format_det_output(detect)
        mAP = self._compute_map(pred_labels, batch)
        self.log('validation/train end mAP50', mAP['map_50'])
        self.log('validation/train end mAP', mAP['map'])

        # add some specific lof for training
        self.log('validation/hmaps_loss', ret['hmaps_loss'])
        self.log('validation/size_loss', ret['size_loss'])
        self.log('validation/off_loss', ret['off_loss'])
        # self.log('validation/loss_det', loss_det, on_step=True, prog_bar=True)

        return ret['loss_det']

    def predict_step(self, batch, batch_idx):
        ret = self._step(batch)
        return ret
    
    def _format_det_output(self, detect):
        pred_labels = []
        for preds in detect:
            pred_labels.append(
                {
                    'labels': preds[2],
                    'boxes': preds[0],
                    'scores': preds[1],
                    'hmaps': preds[3]
                }
            )
        return pred_labels
    
    def _compute_map(self, predictions, batch, debug=False):
        metrics = MeanAveragePrecision(iou_thresholds=[0.5, 0.75], box_format='xyxy', iou_type='bbox')
        
        pred_list = []
        target_list = []
        for idx, (gt, preds) in enumerate(zip(batch['labels'], predictions)):
            pred_list.append(
                dict(
                    boxes = preds['boxes'],
                    scores =  preds['scores'],
                    labels = preds['labels'].to(int)
                )
            )
            target_list.append(
                dict(
                    boxes = gt['boxes'],
                    labels = gt['labels'].detach().cpu().to(int)
                )
            )
        
        metrics.update(pred_list, target_list)
        mAP = metrics.compute()
        if debug:
            from pprint import pprint
            pprint(mAP)

        return mAP
        
    # ----------------------------  LOGGERS ----------------------------------- # 
    def _log_hmaps_to_tensorboard(self, data, title=''):
        for batch_idx, preds in enumerate(data):
            pred_hmpas = preds['hmaps']
            classes_id = preds['labels']

            classes_id = classes_id.to(torch.int64)
            for id in classes_id:
                tensor_hmaps = self._hmaps_to_tensor(batch_idx, pred_hmpas[id], id)
                min_activaiton = round(pred_hmpas[id].min().item(), 3)
                max_activaiton = round(pred_hmpas[id].max().item(), 3)
                self.logger.experiment.add_image(f"{title}hm: batch_idx[{batch_idx}] - classe_name {classes_names_list[id]}, min:{min_activaiton} max:{max_activaiton}", tensor_hmaps, self.current_epoch)

    def _hmaps_to_tensor(self, batch_idx, hmap, associated_id):
        buf, fig = generate_heatmap_to_buf(hmap, title=f"hm: batch_idx {batch_idx} - classe_name {classes_names_list[associated_id]}")
        image = Image.open(buf)
        tensor_image = TF.to_tensor(image)
        buf.close()
        plt.close(fig)
        return tensor_image

    def _annotated_image_to_tensor(self, batch_idx, img, associated_bbox, classes_id):
        fig, buf = print_base_img_annotated_img(img, box_xyxy=associated_bbox, classes_id=classes_id, classes_name_list=classes_names_list, log_tensor=True)
        image = Image.open(buf)
        tensor_image = TF.to_tensor(image)
        buf.close()
        self.logger.experiment.add_image(f"batch_idx == {batch_idx}", tensor_image, self.trainer.current_epoch)
        plt.close(fig)
        return tensor_image

    def _log_image_to_tensorboard(self, images_list, associated_labels_list, need_detach_gpu=False, title=''):
        for idx, (image, lab) in enumerate(zip(images_list, associated_labels_list)):
            classes_id = lab['labels']
            bounding_boxes = lab['boxes']
            
            scores = lab.get("scores") # TODO implement the rest of the process to print the scores

            if need_detach_gpu:
                classes_id = classes_id.detach().cpu()
                bounding_boxes = bounding_boxes.detach().cpu()
                image = image.detach().cpu()


            tensor_image = self._annotated_image_to_tensor(idx, image, bounding_boxes, classes_id=classes_id)
            self.logger.experiment.add_image(f"{title} batch_idx == {idx}", tensor_image, self.trainer.current_epoch)
    
    # ------------------------------------------------------------------------- # 
    




class InputMonitor(pl.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        img = batch['img']
        # labels = batch['labels']
        hmaps = batch['hmaps']
        # off_maps = batch['offset_map']
        # size_maps = batch['size_map']
        # center_position_save  = batch['center_position_save']
        
        logger = trainer.logger
        logger.experiment.add_histogram("input", img, global_step=trainer.global_step)
        logger.experiment.add_histogram("target", hmaps, global_step=trainer.global_step)


if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger("logs", name="CenterNet")

    dm = CardDlayingDataModule(batch_size = 2)

    model = CenterNetpl(    batch_size= 2,
                            input_shape = (3, 256, 256),
                            nclasses = 53,
                            size_loss_weight = 2.5,
                            offset_loss_weight = 1,
                            focal_loss_weight = 1,
                            overfit_mode=True, # compatible only with overfit_batches=1 trainer arg else put False
                            topk=4,
                            activation_treshold = 0.0
                            )

    trainer = pl.Trainer(log_every_n_steps=1, 
                         max_epochs=2,
                         accelerator='mps', 
                         logger=tb_logger, 
                         overfit_batches=1,
                         limit_predict_batches=1,
                         limit_val_batches=1,
                         limit_test_batches=1,
                         fast_dev_run=False, 
                         deterministic=True
                         )

    trainer.fit(model, dm)
    trainer.test(model, dm)



    # ##############################@ SOME DEBUG FOR THE COMPUTE MAP FUNCTION ########################################

    #    print('------ predictions --------')

    #     for idx, pr in enumerate(pred_list):
    #         print('batch_idx:', idx)
    #         print('boxes infos:', type(pr['boxes']), pr['boxes'].shape, pr['boxes'].dtype, '\n', pr['boxes'])
    #         print('scores info:', type(pr['scores']), pr['scores'].shape, pr['scores'].dtype, pr['scores'])
    #         print('labels info:', type(pr['labels']), pr['labels'].shape, pr['labels'].dtype, pr['labels'])

    #     print('\n------ ----------- --------\n')

    #     print('------ ground truth --------')
    #     for idx, pr in enumerate(target_list):
    #         print('batch_idx:', idx)
    #         print('boxes infos:', type(pr['boxes']), pr['boxes'].shape, pr['boxes'].dtype, '\n', pr['boxes'])
    #         print('labels info:', type(pr['labels']), pr['labels'].shape, pr['labels'].dtype, pr['labels'])
        
    #     print('\n------ ------------ --------')

       