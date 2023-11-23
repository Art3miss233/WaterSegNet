import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath('/home/emilia/WaterSegNet'))
from lib.dataloader import SegDataset
import matplotlib.pyplot as plt


def remove_border(mask, original_size):
    h, w = mask.shape
    orig_h, orig_w = original_size
    delta_w = w - orig_w
    delta_h = h - orig_h

    if delta_w > 0:
        mask = mask[:, delta_w // 2 : -delta_w // 2]
    if delta_h > 0:
        mask = mask[delta_h // 2 : -delta_h // 2, :]
    return mask


def predict_image(model, image, device):
    model.eval()
    img = torch.from_numpy(SegDataset.preprocess(image, is_mask=False))
    img = img.unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        pred = model(img).cpu()
        max_dim = max(image.size)
        pred = F.interpolate(pred, size=(max_dim, max_dim))
        mask = torch.sigmoid(pred) > 0.5

    mask_np = mask[0].long().squeeze().numpy()
    mask_np = remove_border(mask_np, original_size=(image.size[1], image.size[0]))
    return mask_np


def plot_prediction(image_raw, model, device, title="Prediction"):
    plt.figure(figsize=(10, 5))
    plt.imshow(image_raw)
    mask = predict_image(model, image_raw, device)
    plt.imshow(mask, alpha=0.5, cmap="Accent")
    plt.title(title)
    plt.axis("off")
    plt.show()

