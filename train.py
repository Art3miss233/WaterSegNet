import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models_pytorch as smp
from lib.dataloader import SegDataset
import datetime
from pathlib import Path
import torchmetrics
from lib.predict import *

class SegModel(LightningModule):
    def __init__(self, model, lr, optimizer_type, train_metrics, val_metrics, freeze_encoder=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.freeze_encoder = freeze_encoder
        self.criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.train_metric_tracker = train_metrics
        self.val_metric_tracker = val_metrics

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        optimizer = {
            "adam": torch.optim.Adam(self.model.parameters(), lr=self.lr),
            "sgd": torch.optim.SGD(self.model.parameters(), lr=self.lr),
            "rmsprop": torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        }.get(self.optimizer_type, None)

        if optimizer is None:
            raise ValueError(f"Unknown optimizer type {self.optimizer_type}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def training_step(self, batch, batch_idx):
        imgs, gt_masks = batch["image"], batch["mask"]
        mask_pred = self(imgs)

        crit_loss = self.criterion(mask_pred.squeeze(1), gt_masks.float())
        dice_loss = self.dice_loss(mask_pred, gt_masks.float())
        loss = crit_loss + dice_loss

        mask_pred = (torch.sigmoid(mask_pred) > 0.5).squeeze(1).long()
        self.train_metric_tracker.update(mask_pred, gt_masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gt_masks = batch["image"], batch["mask"]
        mask_pred = self(imgs)
        crit_loss = self.criterion(mask_pred.squeeze(1), gt_masks.float())
        dice_loss = self.dice_loss(mask_pred.squeeze(1), gt_masks.float())
        loss = crit_loss + dice_loss
    
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).squeeze(1).long()
        
        self.val_metric_tracker.update(mask_pred, gt_masks)
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_metric_tracker.increment()

    def on_train_epoch_start(self):
        self.train_metric_tracker.increment()

    def on_validation_epoch_end(self):
        val_metric_result= self.val_metric_tracker.compute_all()
        for key, value in val_metric_result.items():
            self.log(f"val.{key}", value, on_epoch=True)

    def on_train_epoch_end(self):
        train_metric_result= self.train_metric_tracker.compute_all()
        for key, value in train_metric_result.items():
            self.log(f"train.{key}", value, on_epoch=True)
        

class SegDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=os.cpu_count(), 
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=os.cpu_count(), 
                          pin_memory=True, 
                          drop_last=True, 
                          )

if __name__ == "__main__":
    root_dir = Path(
        "/home/emilia/msc_ros2/master_ws/src/msc_thesis/offline_utils/train_segmentation/custom_dataset/"
    )

    dir_test_img = root_dir / "test/images"
    dir_test_mask = root_dir / "test/labels"
    dir_train_img = root_dir / "train/images"
    dir_train_mask = root_dir / "train/labels"
    dir_valid_img = root_dir / "valid/images"
    dir_valid_mask = root_dir / "valid/labels"

    # Directory paths for checkpoints and best models
    dir_checkpoint = root_dir / "checkpoints/"
    dir_best_model = root_dir / "best_models/"
    dir_best_model /= datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")

    train_set = SegDataset(dir_train_img, dir_train_mask)
    valid_set = SegDataset(dir_valid_img, dir_valid_mask)
    test_set = SegDataset(dir_test_img, dir_test_mask)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    model = model.to(memory_format=torch.channels_last)

    metrics = torchmetrics.MetricCollection(
        torchmetrics.Accuracy(task="binary", num_classes=1, multiclass=False),
        torchmetrics.Recall(task="binary", num_classes=1, multiclass=False),
        torchmetrics.Precision(task="binary", num_classes=1, multiclass=False),
        torchmetrics.F1Score(task="binary", num_classes=1, multiclass=False), # Dice Coefficient
    )

    train_metrics = torchmetrics.MetricTracker(metrics)
    val_metrics = torchmetrics.MetricTracker(metrics)   

    seg_model = SegModel(model, 
                         lr=1e-3, 
                         optimizer_type="rmsprop", 
                         train_metrics=train_metrics, 
                         val_metrics=val_metrics, 
                         freeze_encoder=False
                         )
    data_module = SegDataModule(train_set, valid_set, batch_size=4)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/", 
        save_top_k=1, 
        monitor="val_loss", 
        mode="min", 
        filename="model-{}".format(timestamp)
        )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")


    trainer = Trainer(
        max_epochs=4,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stopping],
        precision=16,
    )

    trainer.fit(seg_model, data_module)


