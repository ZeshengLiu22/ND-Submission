import torch
import torchvision.models as models
import matplotlib.pylab as plt

from PIL import Image
from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *

import os
from tqdm import tqdm

# ++++++++++++++++++++ Cityscapes Visualisation +++++++++++++++++++++++++
def vis_cityscapes():
    data_path = 'dataset/cityscapes'
    im_size = [512, 1024]
    num_segments = 19
    test_idx = get_cityscapes_idx(data_path, train=False)

    device = torch.device("cpu")
    model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to(device)
    colormap = create_cityscapes_label_colormap()

    # visualise image id 61 in validation set
    im_id = 61
    im = Image.open('dataset/cityscapes/images/val/{}.png'.format(im_id))
    gt_label = Image.fromarray(cityscapes_class_map(Image.open('dataset/cityscapes/labels/val/{}.png'.format(im_id))))
    im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0),augmentation=False)


    model.load_state_dict(torch.load('model_weights/cityscapes_label20_sup.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_sup = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_sup[label_tensor == -1] = -1

    model.load_state_dict(torch.load('model_weights/cityscapes_label20_semi_classmix.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_classmix = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_classmix[label_tensor == -1] = -1

    model.load_state_dict(torch.load('model_weights/cityscapes_label20_semi_classmix_reco.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_classmix[label_tensor == -1] = -1

    fig, ax = plt.subplots(1, 4, figsize=(10, 6))

    gt_blend = Image.blend(im, Image.fromarray(color_map(label_tensor[0].numpy(), colormap)), alpha=.7)
    sup_blend = Image.blend(im, Image.fromarray(color_map(label_sup[0].numpy(), colormap)), alpha=.7)
    classmix_blend = Image.blend(im, Image.fromarray(color_map(label_classmix[0].numpy(), colormap)), alpha=.7)
    reco_blend = Image.blend(im, Image.fromarray(color_map(label_reco[0].numpy(), colormap)), alpha=.7)

    ax[0].imshow(gt_blend)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xlabel('Ground Truth')
    ax[1].imshow(sup_blend)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('Supervised')
    ax[2].imshow(classmix_blend)
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xlabel('ClassMix')
    ax[3].imshow(reco_blend)
    ax[3].set_xticklabels([])
    ax[3].set_yticklabels([])
    ax[3].set_xlabel('ClassMix + ReCo')

def vis_pascal():
# ++++++++++++++++++++ Pascal VOC Visualisation +++++++++++++++++++++++++
    im_size = [513, 513]
    root = 'dataset/pascal'
    with open(root + '/val.txt') as f:
        idx_list = f.read().splitlines()

    num_segments = 21
    device = torch.device("cpu")
    model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to(device)
    colormap = create_pascal_label_colormap()

    # visualise image id 961 in validation set
    id = 961
    im_id = idx_list[id]
    im = Image.open(root + '/JPEGImages/{}.jpg'.format(im_id))
    gt_label = Image.open(root + '/SegmentationClassAug/{}.png'.format(im_id))
    im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)
    im_w, im_h = im.size

    model.load_state_dict(torch.load('model_weights/pascal_label60_sup.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_sup = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_sup[label_tensor == -1] = -1

    model.load_state_dict(torch.load('model_weights/pascal_label60_semi_classmix.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_classmix = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_classmix[label_tensor == -1] = -1

    model.load_state_dict(torch.load('model_weights/pascal_label60_semi_classmix_reco.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, rep = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_reco[label_tensor == -1] = -1

    fig, ax = plt.subplots(1, 4, figsize=(10, 6))

    gt_blend = Image.blend(im, Image.fromarray(color_map(label_tensor[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
    sup_blend = Image.blend(im, Image.fromarray(color_map(label_sup[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
    classmix_blend = Image.blend(im, Image.fromarray(color_map(label_classmix[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
    reco_blend = Image.blend(im, Image.fromarray(color_map(label_reco[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)

    ax[0].imshow(gt_blend)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xlabel('Ground Truth')
    ax[1].imshow(sup_blend)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('Supervised')
    ax[2].imshow(classmix_blend)
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xlabel('ClassMix')
    ax[3].imshow(reco_blend)
    ax[3].set_xticklabels([])
    ax[3].set_yticklabels([])
    ax[3].set_xlabel('ClassMix + ReCo')

def vis_rescuenet(idx):
    im_size = [3000, 4000]
    root = 'dataset/rescuenet'

    num_segments = 11
    device = torch.device("cpu")
    model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to(device)
    colormap = create_rescuenet_label_colormap()

    # visualise image
    im_id = idx
    im = Image.open(root + '/validationset/val-org-img/{}.jpg'.format(im_id))
    gt_label = Image.open(root + '/validationset/val-label-img/{}_lab.png'.format(im_id))
    im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)
    im_w, im_h = im.size

    model.load_state_dict(torch.load('model_weights/rescuenet_label15_semi_classmix_reco_0.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, rep = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_reco[label_tensor == -1] = -1

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))

    gt_output = Image.fromarray(color_map(label_tensor[0].numpy(), colormap)[:im_h, :im_w])
    reco_output = Image.fromarray(color_map(label_reco[0].numpy(), colormap)[:im_h, :im_w])

    ax[0].imshow(im)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xlabel('Raw Image')
    ax[1].imshow(gt_output)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('Ground Truth')
    ax[2].imshow(reco_output)
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xlabel('ReCo')

    filename = f'{idx}_reco_rescuenet.png'
    fig.savefig(filename)

def vis_floodnet(idx):
    im_size = [3000, 4000]
    root = 'dataset/floodnet'

    num_segments = 10
    device = torch.device("cpu")
    model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to(device)
    colormap = create_floodnet_label_colormap()

    # visualise image
    im_id = idx
    im = Image.open(root + '/validationset/val-org-img/{}.jpg'.format(im_id))
    gt_label = Image.open(root + '/validationset/val-label-img/{}_lab.png'.format(im_id))
    im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)
    im_w, im_h = im.size

    model.load_state_dict(torch.load('model_weights/floodnet_label15_semi_classmix_reco_0.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, rep = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_reco[label_tensor == -1] = -1

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))

    gt_output = Image.fromarray(color_map(label_tensor[0].numpy(), colormap)[:im_h, :im_w])
    reco_output = Image.fromarray(color_map(label_reco[0].numpy(), colormap)[:im_h, :im_w])

    ax[0].imshow(im)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xlabel('Raw Image')
    ax[1].imshow(gt_output)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('Ground Truth')
    ax[2].imshow(reco_output)
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xlabel('Wrong Class')

    filename = f'new_{idx}_floodnet.png'
    
    fig.savefig(filename)

if __name__ == "__main__":
    for index in [7468, 7607, 7608]:
        vis_floodnet(index)