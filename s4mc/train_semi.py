import argparse
import logging
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from s4mc_utils.dataset.augmentation import generate_unsup_data
from s4mc_utils.dataset.builder import get_loader
from s4mc_utils.models.model_helper import ModelBuilder
from s4mc_utils.utils.dist_helper import setup_distributed
from s4mc_utils.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
from s4mc_utils.utils.visual_utils import visual_evaluation
from s4mc_utils.utils.lr_helper import get_optimizer, get_scheduler
from s4mc_utils.utils.utils import (
    AverageMeter,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)
#for AEL run
#from s4mc_utils.utils.utils import (dynamic_copy_paste,update_cutmix_bank,generate_cutmix_mask,cal_category_confidence,sample_from_bank)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--name", default="regular")
parser.add_argument("--mode", type=str, default="train")

def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        tb_logger = SummaryWriter(osp.join(cfg["exp_path"], f"log/events_seg/{args.name}_{seed}"))
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)
    
    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]


    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    #If you run with u2pl or ael:
    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    q=cfg["trainer"]["contrastive"]["num_queries"]
    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros((cfg["net"]["num_classes"],q,1,256,)).cuda()

    # Start to train model
    if cfg["main_mode"].get("eval", False):
        #uncomment this line if you want to use the refine_infer function
        #prec = refine_infer(model_teacher, val_loader, 0, logger) 
        prec = validate(model_teacher, val_loader, 0, logger) 
        if rank ==0:
            logger.info("evaluation result: {}".format(prec))
    elif cfg["main_mode"].get("compare", False):
        # Teacher model
        base_model = ModelBuilder(cfg["net"])
        base_model = base_model.cuda()
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        for p in base_model.parameters():
            p.requires_grad = False

        load_state(cfg["main_mode"]["compared_pretrain"], base_model, key="teacher_state")
        if rank == 0:
            logger.info("start comparison")
        compare(base_model,model_teacher, val_loader, logger)
    else:
        for epoch in range(last_epoch, cfg_trainer["epochs"]):
            # Training
            train(
                model,
                model_teacher,
                optimizer,
                lr_scheduler,
                sup_loss_fn,
                train_loader_sup,
                train_loader_unsup,
                epoch,
                tb_logger,
                logger,
                memobank,
                queue_ptrlis,
                queue_size,)

            # Validation
            if cfg_trainer["eval_on"]:
                if rank == 0:
                    logger.info("start evaluation")

                if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                    prec = validate(model, val_loader, epoch, logger)
                else:
                    prec = validate(model_teacher, val_loader, epoch, logger)

                if rank == 0:
                    state = {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "teacher_state": model_teacher.state_dict(),
                        "best_miou": best_prec,
                    }
                    if not os.path.exists(osp.join(cfg["saver"]["snapshot_dir"], args.name)):
                        os.makedirs(osp.join(cfg["saver"]["snapshot_dir"], args.name))

                    if prec > best_prec:
                        best_prec = prec
                        torch.save(
                            state, osp.join(osp.join(cfg["saver"]["snapshot_dir"], args.name),"ckpt_best.pth")
                        )

                    torch.save(state, osp.join(osp.join(cfg["saver"]["snapshot_dir"], args.name), "ckpt.pth"))

                    logger.info(
                        "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                            best_prec * 100
                        )
                    )
                    tb_logger.add_scalar("mIoU val", prec, epoch)


def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    memobank,
    queue_ptrlis,
    queue_size,
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]
    model.train()

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    B_ratio= len(loader_l) / len(loader_u)

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)


    batch_end = time.time()
    for step in range(1): #len(loader_l)):
        batch_start = time.time()

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = next(loader_l_iter)
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, _ = next(loader_u_iter)
        image_u = image_u.cuda()
        
        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs = model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )


            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )
                
                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )

            # unsupervised loss
            drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
            drop_percent = 100 - percent_unreliable
            indicator = cfg["trainer"]["unsupervised"].get("indicator", "margin")
            neigborhood_size = cfg["trainer"]["unsupervised"].get("neigborhood_size", 4)
            n_neigbors = cfg["trainer"]["unsupervised"].get("n_neigbors", 1)
            ds = "pascal" if "pascal" in cfg["dataset"]["type"] else "cityscapes"

            unsup_loss = compute_unsupervised_loss(
                        pred_u_large,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher.detach(),
                        indicator,
                        ds,
                        neigborhood_size,
                        n_neigbors) * cfg["trainer"]["unsupervised"].get("unsup_weight", 1) * B_ratio
            
            

            # contrastive for using S4MC+U2PL
            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False):
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                    1 - epoch / cfg["trainer"]["epochs"]
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )
                    # down sample

                    if cfg_contra.get("negative_high_entropy", True):
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_aug.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(label_l, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )

                if cfg_contra.get("binary", False):
                    contra_flag += " BCE"
                    contra_loss = compute_binary_memobank_loss(
                        rep_all,
                        torch.cat((label_l_small, label_u_small)).long(),
                        low_mask_all,
                        high_mask_all,
                        prob_all_teacher.detach(),
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    if not cfg_contra.get("anchor_ema", False):
                        new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                        )
                    else:
                        prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            prototype,
                        )

                dist.all_reduce(contra_loss)
                contra_loss  = contra_loss/ world_size

            else:
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss #* 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)




        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}][{}] "
                "Iter [{}/{}]  "
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})  "
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})  "
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})  "
                "u2pl {con_loss.avg:.3f} "
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    'S4MC',
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)

    return


def compare(base_model,model,data_loader,logger):
    base_model.eval()
    model.eval()
    data_loader.sampler.set_epoch(0)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)
            base_out= base_model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(output, labels.shape[1:], mode="bilinear", align_corners=True)
        output = output.data.max(1)[1]#.cpu().numpy()

        base_output = base_out["pred"]
        base_output = F.interpolate(base_output, labels.shape[1:], mode="bilinear", align_corners=True)
        base_output = base_output.data.max(1)[1]#.cpu().numpy()
        
        labels[labels == 255] = 0
        if step==0:
            _images=images
            _labels=labels
            _output=output
            _base_output=base_output
        else:
            _images=torch.cat((_images,images),0)
            _labels=torch.cat((_labels,labels),0)
            _output=torch.cat((_output,output),0)
            _base_output=torch.cat((_base_output,base_output),0)
        #if step==4:
        #    break
        
    torch.save(_images.cpu(),'imgs/images_'+str(rank)+'.pt')
    torch.save(_labels.cpu(),'imgs/labels_'+str(rank)+'.pt')
    torch.save(_output.cpu(),'imgs/output_'+str(rank)+'.pt')
    torch.save(_base_output.cpu(),'imgs/base_output_'+str(rank)+'.pt')

    dist.all_reduce(labels)

    if rank == 0:
        logger.info(" *done comparing* ")
    return


def save_for_heatmap(model,data_loader,i):
    
    model.eval()
    data_loader.sampler.set_epoch(0)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank= dist.get_rank()
    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        output = outs["pred"]
        output = F.interpolate(output, labels.shape[1:], mode="bilinear", align_corners=True)
        #output = output.data.max(1)[1]#.cpu().numpy()

        if step==0:
            _output=output
        else:
            _output=torch.cat((_output,output),0)
        if step==5:
            break
        
    dist.all_reduce(output)
    torch.save(_output.cpu(),'imgs/hm/'+str(i)+'output.pt')
    
    return


def do_all_gather(tensor, size): 
    tensor_out_list = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(tensor_out_list, tensor)
    out_tensor = torch.cat(tensor_out_list)
    return out_tensor


def refine_infer(
    model,
    data_loader,
    epoch,
    logger,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
    
        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )

        prob = F.softmax(output.data, dim=1)

        ext = torch.nn.functional.pad(torch.clone(prob), (1,1,1,1,0,0,0,0))
        left = ext[:,:,:-2, 1:-1]
        right = ext[:,:,2:,1:-1]
        up = ext[:,:,1:-1,:-2]
        down = ext[:,:,1:-1, 2:]
        d=torch.stack((left, right, up, down)).to(prob.device)
        arr,neigbor_idx=torch.max(d,0) 
        beta = torch.exp(torch.tensor(-1/2)) #for more neigbors use neigbor_idx 
        


        ext = torch.nn.functional.pad(torch.clone(prob), (2,2,2,2,0,0,0,0))
        left = ext[:,:,:-4, 2:-2]
        right = ext[:,:,4:,2:-2]
        up = ext[:,:,2:-2,:-4]
        down = ext[:,:,2:-2, 4:]
        d=torch.stack((left, right, up, down)).to(prob.device)
        arr2,neigbor_idx=torch.max(d,0)
        
        ext = torch.nn.functional.pad(torch.clone(prob), (4,4,4,4,0,0,0,0))
        left = ext[:,:,:-8, 4:-4]
        right = ext[:,:,8:,4:-4]
        up = ext[:,:,4:-4,:-8]
        down = ext[:,:,4:-4, 8:]
        d=torch.stack((left, right, up, down)).to(prob.device)
        arr3,neigbor_idx=torch.max(d,0)

        ext = torch.nn.functional.pad(torch.clone(prob), (5,5,5,5,0,0,0,0))
        left = ext[:,:,:-10, 5:-5]
        right = ext[:,:,10:,5:-5]
        up = ext[:,:,5:-5,:-10]
        down = ext[:,:,5:-5, 10:]
        d=torch.stack((left, right, up, down)).to(prob.device)
        arr4,neigbor_idx=torch.max(d,0)
        
        prob = prob + beta*arr - (prob*arr*beta)
        #prob = prob + beta*arr2 - (prob*arr2*beta)
        #prob = prob + beta*arr3 - (prob*arr3*beta)
        prob = prob + beta*arr4 - (prob*arr4*beta)


        output = prob.max(1)[1].cpu().numpy()


        #output = output.data.max(1)[1].cpu().numpy()
                
        target_origin = labels.cpu().numpy() #shape
        
        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )
        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


def validate(
    model,
    data_loader,
    epoch,
    logger,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for step, batch in enumerate(data_loader):
        images, labels, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
    
        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        """print('------------')
        label_masks_tuple = F.one_hot(labels).chunk(num_classes,dim=-1)
        output_masks_tuple = output.chunk(num_classes,dim=1)
        for i in range(len(label_masks_tuple)):
            print(label_masks_tuple[i].squeeze().cpu().numpy().shape)
            label_boundary = mask_to_boundary(label_masks_tuple[i].squeeze().cpu().numpy())
            output_boundary = mask_to_boundary(output_masks_tuple[i].squeeze().cpu().numpy()) 
            intersection = output_boundary[np.where(output_boundary == label_boundary)].sum()
            area_output= output_boundary[np.where(output_boundary == 1)].sum()
            area_target = label_boundary[np.where(label_boundary == 1)].sum()
            area_union = area_output + area_target - intersection
            print(intersection, area_union, area_target)
        print('------------')
        exit(0)"""

        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy() #shape
        
        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )
        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU

def sort_array_by_array(array, array_to_sort):
    return array[np.argsort(array_to_sort)]


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def get_expectation(trani_loader,classes = 19):
    #or 21
    counter,totals = [0]*classes,[0]*classes
    loader_iter = iter(trani_loader)
    for step in range(len(trani_loader)):
        _, label = next(loader_iter)
        batch_size, h, w = label.size()
        #outs = model(image)
        #pred, rep = outs["pred"], outs["rep"]
        #pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)
        ext = torch.nn.functional.pad(label, (1, 1, 1, 1, 0, 0))
        left = ext[:,:-2, 1:-1]
        right = ext[:,2:,1:-1]
        up = ext[:,1:-1,:-2]
        down = ext[:,1:-1, 2:]
        d=torch.stack((left, right, up, down))
        def expand(l,classes=19):
            if len(l) !=classes:
                return l+[0]*(classes-len(l))
            return l
        _totals = expand(list(label.reshape(-1).bincount().numpy()))
        totals = [a + b for a, b in zip(totals, _totals)]
        for neigbor in d:
            inds=torch.where(neigbor==label)
            _counter = expand(list(label[inds].bincount().numpy()))
            counter = [a + b for a, b in zip(counter, _counter)]
    probs = [a/(4*b) for a,b in zip(counter,totals)]
    print(probs)
    return probs


if __name__ == "__main__":
    main()