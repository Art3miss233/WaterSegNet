import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models_pytorch as smp
from lib.dataloader import SegDataset, SegDataModule
import datetime
from pathlib import Path
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger

class SegModel(LightningModule):
    def __init__(self, 
                 model, 
                 lr, 
                 optimizer_type, 
                 train_metrics, 
                 val_metrics, 
                 test_metrics, 
                 freeze_encoder=False
                 ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.freeze_encoder = freeze_encoder
        self.criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.train_metric_tracker = train_metrics
        self.val_metric_tracker = val_metrics
        self.test_metric_tracker = test_metrics

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        optimizer = {
            "adam": torch.optim.Adam(self.model.parameters(), lr=self.lr),
            "adamw": torch.optim.AdamW(self.model.parameters(), lr=self.lr),
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
        dice_loss = self.dice_loss(mask_pred, gt_masks.float())
        loss = crit_loss + dice_loss
    
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).squeeze(1).long()
        
        self.val_metric_tracker.update(mask_pred, gt_masks)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        imgs, gt_masks = batch["image"], batch["mask"]

        mask_pred = self(imgs)
        crit_loss = self.criterion(mask_pred.squeeze(1), gt_masks.float())
        dice_loss = self.dice_loss(mask_pred, gt_masks.float())
        loss = crit_loss + dice_loss
    
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).squeeze(1).long()
        
        self.test_metric_tracker.update(mask_pred, gt_masks)
        self.log("test_loss", loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_metric_tracker.increment()

    def on_train_epoch_start(self):
        self.train_metric_tracker.increment()

    def on_test_epoch_start(self):
        self.test_metric_tracker.increment()

    def on_validation_epoch_end(self):
        val_metric_result= self.val_metric_tracker.compute_all()
        for key, value in val_metric_result.items():
            self.log(f"val.{key}", value, on_epoch=True)

    def on_train_epoch_end(self):
        train_metric_result= self.train_metric_tracker.compute_all()
        for key, value in train_metric_result.items():
            self.log(f"train.{key}", value, on_epoch=True)
    
    def on_test_epoch_end(self):
        test_metric_result= self.test_metric_tracker.compute_all()
        for key, value in test_metric_result.items():
            self.log(f"test.{key}", value, on_epoch=True)



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
        torchmetrics.F1Score(
            task="binary", num_classes=1, multiclass=False
        ),  # Dice Coefficient
    )

    train_metrics = torchmetrics.MetricTracker(metrics)
    val_metrics = torchmetrics.MetricTracker(metrics)
    test_metrics = torchmetrics.MetricTracker(metrics)


    seg_model = SegModel(
        model,
        lr=1e-3,
        optimizer_type="adam",
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        freeze_encoder=False,
    )
    data_module = SegDataModule(train_set, valid_set, test_set, batch_size=16)
    model_name = "unet_resnet34_adam_padded_b16"

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="model-{}".format(model_name),
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, verbose=True, mode="min"
    )

    tb_logger = TensorBoardLogger("lightning_logs/", name=model_name)

    trainer = Trainer(
        max_epochs=200,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        precision=16,  # Mixed precision training
    )

    # trainer.fit(seg_model, data_module)
    trainer.test(
        seg_model,
        ckpt_path="checkpoints/model-unet_resnet34_adam_padded_b16.ckpt",
        datamodule=data_module,
        verbose=True,
    )


