import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from seg_core.model import MiT_SegFormer
from seg_core import eval_seg


def val(model, data_loader, num_classes):
    model.eval()
    preds, gts = [], []
    for data in data_loader:
        with torch.no_grad():
            image = data['img'][0].cuda(non_blocking=True)
            label = data['gt_semantic_seg'][0].cuda(non_blocking=True)
            output = model(image)
            output = F.interpolate(output, size=label.shape[1:], mode='bilinear', align_corners=False)
            preds += list(torch.argmax(output, dim=1).cpu().numpy().astype(np.int16))
            gts += list(label.cpu().numpy().astype(np.int16))
    score = eval_seg.scores(gts, preds, num_classes=num_classes)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='mit_b1')
    parser.add_argument('--num-classes', type=int, default=11)
    args = parser.parse_args()

    # Load config and dataset
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        pin_memory=True
    )

    # Load model
    model = MiT_SegFormer(backbone=args.backbone,
                          num_classes=args.num_classes,
                          embedding_dim=256,
                          pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model = model.cuda()
    model.eval()

    # Run validation
    score = val(model, dataloader, args.num_classes)

    print("\n==== Evaluation Results ====")
    for class_id, iou in score['Class IoU'].items():
        print(f"Class {class_id}: IoU = {iou:.4f}")
    print(f"Mean IoU: {score['Mean IoU'] * 100:.2f}")
    print(f"FWIoU:    {score['FWIoU'] * 100:.2f}")


if __name__ == '__main__':
    main()
