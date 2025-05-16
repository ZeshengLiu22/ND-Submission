import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

from s4mc_utils.models.model_helper import ModelBuilder
from s4mc_utils.dataset.builder import get_loader
from s4mc_utils.utils.utils import intersectionAndUnion, convert_state_dict


def get_rescuenet_palette():
    return [
        0, 0, 0,         # unlabeled
        61, 230, 250,    # water
        180, 120, 120,   # building-no-damage
        235, 255, 7,     # building-medium-damage
        255, 184, 6,     # building-major-damage
        255, 0, 0,       # building-total-destruction
        255, 0, 245,     # vehicle
        140, 140, 140,   # road-clear
        160, 150, 20,    # road-blocked
        4, 250, 7,       # tree
        255, 235, 0      # pool
    ]


def save_prediction(mask, name, color_folder, palette):
    os.makedirs(color_folder, exist_ok=True)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape: {mask.shape}")

    color_mask = Image.fromarray(mask.astype(np.uint8), mode="P")
    color_mask.putpalette(palette)
    color_mask.save(os.path.join(color_folder, name + ".png"))


@torch.no_grad()
def evaluate(model, loader, cfg, save_dir=None):
    model.eval()
    num_classes = cfg["net"]["num_classes"]
    ignore_label = cfg["dataset"]["ignore_label"]

    color_dir = os.path.join(save_dir, "color") if save_dir else None
    palette = get_rescuenet_palette() if save_dir else None

    inter_meter = np.zeros(num_classes)
    union_meter = np.zeros(num_classes)
    target_meter = np.zeros(num_classes)
    per_image_ious = []

    for _, (images, labels, names) in enumerate(tqdm(loader)):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)["pred"]
        preds = F.interpolate(preds, size=labels.shape[1:], mode="bilinear", align_corners=True)
        preds = preds.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        for i in range(preds.shape[0]):
            pred_i = preds[i]
            label_i = labels[i]

            if save_dir:
                save_prediction(pred_i, names[i], color_dir, palette)

            inter, union, target = intersectionAndUnion(pred_i, label_i, num_classes, ignore_label)
            inter_meter += inter
            union_meter += union
            target_meter += target

            iou = inter / (union + 1e-10)
            per_image_ious.append(np.mean(iou))

    iou = inter_meter / (union_meter + 1e-10)
    acc = inter_meter / (target_meter + 1e-10)
    miou = np.mean(iou)
    acc_avg = np.mean(acc)
    fwiou = (target_meter * iou).sum() / (target_meter.sum() + 1e-10)

    print("\n--- Evaluation Results ---")
    for i in range(num_classes):
        print(f"Class {i:02d} | IoU: {iou[i]*100:.2f}% | Acc: {acc[i]*100:.2f}%")
    print(f"\nmIoU: {miou*100:.2f}%, Acc: {acc_avg*100:.2f}%, fwIoU: {fwiou*100:.2f}%")

    per_image_ious = np.array(per_image_ious)
    print(f"\nPer-Image mIoU: mean = {per_image_ious.mean()*100:.2f}%, std = {per_image_ious.std()*100:.2f}%")

    return miou, acc_avg, fwiou


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional output dir for visual results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model = ModelBuilder(cfg["net"])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    key = "teacher_state" if "teacher_state" in ckpt else "model_state"
    model.load_state_dict(convert_state_dict(ckpt[key]), strict=False)
    model.cuda()

    _, _, val_loader = get_loader(cfg, seed=0)
    evaluate(model, val_loader, cfg, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
