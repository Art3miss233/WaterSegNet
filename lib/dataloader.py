from torch.utils.data import Dataset
from PIL import Image
import logging
import numpy as np
import torch

from pathlib import Path

def add_borders(img):
    # TODO add borders to images
    pass


class SegDataset(Dataset):
    def __init__(self, img_dir: Path, gt_dir: Path):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.ids = [
            entry.stem
            for entry in img_dir.iterdir()
            if entry.is_file() and not entry.stem.startswith(".")
        ]

        if not self.ids:
            raise RuntimeError("No images found in {}".format(img_dir))

        logging.info(f"Found {len(self.ids)} images in {img_dir}")

    def __getitem__(self, index):
        item = self.ids[index]
        img_path = self.img_dir / (item + ".png")
        gt_path = self.gt_dir / (item + ".png")

        if not img_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(
                f"Image or GT not found for {item} in {self.img_dir} or {self.gt_dir}"
            )

        img = Image.open(img_path)
        gt = Image.open(gt_path)

        
        img = self.preprocess(img, is_mask=False)
        gt = self.preprocess(gt, is_mask=True)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(gt.copy()).long().contiguous(),
        }

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, is_mask, target_res=256):
        resized_img = pil_img.resize(
            (target_res, target_res),
            resample=Image.NEAREST if is_mask else Image.BICUBIC,
        )
        img = np.asarray(resized_img)

        if is_mask:
            mask = np.zeros((target_res, target_res), dtype=np.int64)
            mask[img > 0] = 1
            return mask
        else:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0  # normalize to [0,1]
            return img
