import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from seg_core.model import MiT_SegFormer


def create_rescuenet_label_colormap():
    """Creates a label colormap used in RescueNet segmentation benchmark."""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # Background
    colormap[1] = [61, 230, 250]  # Water
    colormap[2] = [180, 120, 120]  # Building no damage
    colormap[3] = [235, 255, 7]  # Building medium damage
    colormap[4] = [255, 184, 6]  # Building major damage
    colormap[5] = [255, 0, 0]  # Building total destruction
    colormap[6] = [255, 0, 245]  # Vehicle
    colormap[7] = [140, 140, 140]  # Road clear
    colormap[8] = [160, 150, 20]  # Road blocked
    colormap[9] = [4, 250, 7]  # Tree
    colormap[10] = [255, 235, 0]  # Pool
    return colormap


def load_image(path):
    img = Image.open(path).convert('RGB')
    orig_size = img.size  # (W, H)
    w, h = orig_size
    img_small = img.resize((w // 2, h // 2), Image.BILINEAR)
    return transforms.ToTensor()(img_small).unsqueeze(0), (h, w)  # return original H, W


def load_mask(path):
    return np.array(Image.open(path), dtype=np.uint8)


def colorize_mask(mask, colormap):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id, color in enumerate(colormap):
        color_mask[mask == label_id] = color
    return color_mask


def intersection_and_union(pred, target, num_classes, ignore_index=255):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-img-dir', type=str, required=True)
    parser.add_argument('--val-mask-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='./predictions')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='b1')
    parser.add_argument('--num-classes', type=int, default=11)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    colormap = create_rescuenet_label_colormap()

    # Load model on GPU
    model = MiT_SegFormer(backbone='mit_' + args.backbone,
                          num_classes=args.num_classes,
                          embedding_dim=256,
                          pretrained=False)

    # Strip "module." prefix if needed
    raw_state = torch.load(args.checkpoint, map_location='cpu')
    new_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
    model.load_state_dict(new_state)
    model.cuda().eval()

    image_names = sorted(os.listdir(args.val_img_dir))
    total_inter = np.zeros(args.num_classes)
    total_union = np.zeros(args.num_classes)
    total_target = np.zeros(args.num_classes)

    for name in tqdm(image_names, desc='Evaluating'):
        img_path = os.path.join(args.val_img_dir, name)

        base_name = os.path.splitext(name)[0]
        mask_name = base_name + '_lab.png'
        mask_path = os.path.join(args.val_mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Warning: GT mask not found for {name} → {mask_name}, skipping.")
            continue

        img, orig_size = load_image(img_path)  # shape (1, 3, H/2, W/2), original (H, W)
        img = img.cuda()
        gt_mask = load_mask(mask_path)

        with torch.no_grad():
            pred_logits = model(img)
            pred_logits = F.interpolate(pred_logits, size=orig_size, mode='bilinear', align_corners=False)
            pred_mask = pred_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        color_mask = colorize_mask(pred_mask, colormap)
        save_path = os.path.join(args.save_dir, base_name + '.png')
        Image.fromarray(color_mask).save(save_path)

        inter, union, target = intersection_and_union(pred_mask, gt_mask, args.num_classes)
        total_inter += inter
        total_union += union
        total_target += target

    iou = total_inter / (total_union + 1e-10)
    miou = np.nanmean(iou)
    fwiou = (total_target * iou).sum() / (total_target.sum() + 1e-10)

    print("\n==== RescueNet Evaluation (GPU, 2× Downscaled) ====")
    for i, class_iou in enumerate(iou):
        print(f"Class {i}: IoU = {class_iou:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"FWIoU:    {fwiou:.4f}")


if __name__ == '__main__':
    main()
