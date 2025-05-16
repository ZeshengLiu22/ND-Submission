import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from network.deeplabv3.deeplabv3 import DeepLabv3Plus
from build_data import transform, create_floodnet_label_colormap, create_rescuenet_label_colormap


def intersection_and_union(pred, target, num_classes, ignore_index=-1):
    pred = pred.flatten()
    target = target.flatten()
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    intersection = pred[pred == target]
    area_intersection = np.bincount(intersection, minlength=num_classes)
    area_pred = np.bincount(pred, minlength=num_classes)
    area_target = np.bincount(target, minlength=num_classes)
    area_union = area_pred + area_target - area_intersection
    return area_intersection, area_union, area_target


def color_map(label, colormap):
    """Apply colormap to label mask."""
    return colormap[label]


def evaluate_dataset(dataset_name, val_list_file, root_dir, model_path, colormap_fn, num_classes):
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DeepLabv3Plus(torchvision.models.resnet101(pretrained=False), num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    colormap = colormap_fn()
    im_size = [3000, 4000]

    with open(val_list_file) as f:
        idx_list = f.read().splitlines()

    os.makedirs(f"results/{dataset_name}", exist_ok=True)

    metrics = {
        'total_inter': np.zeros(num_classes),
        'total_union': np.zeros(num_classes),
        'total_target': np.zeros(num_classes),
        'per_image_ious': []
    }

    for idx in tqdm(idx_list):
        # Load image and label
        im = Image.open(f'{root_dir}/validationset/val-org-img/{idx}.jpg')
        gt = Image.open(f'{root_dir}/validationset/val-label-img/{idx}_lab.png')
        im_tensor, gt_tensor = transform(im, gt, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)

        im_tensor = im_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        im_w, im_h = im.size

        # Inference
        with torch.no_grad():
            logits, _ = model(im_tensor.unsqueeze(0))
            logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        # Metrics
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt_tensor.squeeze().cpu().numpy()
        inter, union, target = intersection_and_union(pred_np, gt_np, num_classes)
        metrics['total_inter'] += inter
        metrics['total_union'] += union
        metrics['total_target'] += target
        metrics['per_image_ious'].append(np.nanmean(inter / (union + 1e-10)))

        # Save colored mask
        colored_mask = Image.fromarray(color_map(pred_np, colormap)[:im_h, :im_w])
        colored_mask.save(f"results/{dataset_name}/{idx}_mask_colored.png")

    # Final statistics
    class_iou = metrics['total_inter'] / (metrics['total_union'] + 1e-10)
    miou = np.nanmean(class_iou)
    fwiou = np.sum(metrics['total_target'] * class_iou) / (np.sum(metrics['total_target']) + 1e-10)
    per_image_mean = np.mean(metrics['per_image_ious'])
    per_image_std = np.std(metrics['per_image_ious'])

    print(f"\n==== {dataset_name.upper()} RESULTS ====")
    for i, val in enumerate(class_iou):
        print(f"Class {i}: IoU = {val:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"FWIoU: {fwiou:.4f}")
    print(f"Per-image mIoU: Mean = {per_image_mean:.4f}, Std = {per_image_std:.4f}")


if __name__ == "__main__":
    # Evaluate FloodNet
    evaluate_dataset(
        dataset_name='floodnet',
        val_list_file='dataset/floodnet/val.txt',
        root_dir='dataset/floodnet',
        model_path='model_weights/floodnet_reco.pth',
        colormap_fn=create_floodnet_label_colormap,
        num_classes=10
    )

    # Evaluate RescueNet
    evaluate_dataset(
        dataset_name='rescuenet',
        val_list_file='dataset/rescuenet/val.txt',
        root_dir='dataset/rescuenet',
        model_path='model_weights/rescuenet_reco.pth',
        colormap_fn=create_rescuenet_label_colormap,
        num_classes=11
    )
