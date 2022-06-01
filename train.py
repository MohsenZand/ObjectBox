from absl import flags, app
import logging
import math
import os
import sys
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # ObjectBox root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import val  # for end-of-epoch mAP
from utils import attempt_load
from datasets import create_dataloader
from model import Model
from loss import ComputeLoss
from utils import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_img_size, \
    check_file, check_suffix, set_logging, one_cycle, colorstr, methods
from plots import plot_labels
from utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, torch_distributed_zero_first
from loggers import check_wandb_resume
from metrics import fitness
from loggers import Loggers
from callbacks import Callbacks
from flag_sets import *

logger = logging.getLogger(__name__)


def train(hyp, device, callbacks):

    save_dir, epochs, batch_size, weights, single_cls, data, cfg, resume, noval, nosave, workers, freeze, RANK, = \
        Path(FLAGS.save_dir), FLAGS.epochs, FLAGS.batch_size, FLAGS.weights, FLAGS.single_cls, FLAGS.data, FLAGS.cfg, \
        FLAGS.resume, FLAGS.noval, FLAGS.nosave, FLAGS.workers, FLAGS.freeze, FLAGS.global_rank

    # Directories
    wdir = save_dir / 'weights'
    (wdir).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'flags.yaml', 'w') as f:
        #yaml.safe_dump(vars(FLAGS), f, sort_keys=False)
        FLAGS.append_flags_into_file(save_dir / 'flags.yaml')
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        if FLAGS.WANDB:
            include_loggers = ('csv', 'tb', 'wandb')
        else:
            include_loggers = ('csv', 'tb')
        loggers = Loggers(save_dir, weights, FLAGS, hyp, logger, include=include_loggers)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = FLAGS.weights, FLAGS.epochs, FLAGS.hyp

        # Register actions
        for k in methods(loggers):
            if k != 'opt':
                callbacks.register_action(k, callback=getattr(loggers, k))

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        ''' NA for yolov5
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        '''
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc).to(device)  # create

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if FLAGS.adam:
        optimizer = optim.Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if FLAGS.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            #####optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            #####ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            
            #####
            for i in range(len(ema.ema.state_dict().items())):
                if i < 786:
                    list(ema.ema.state_dict().items())[i][1].data = list(ckpt['ema'].float().state_dict().items())[i][1].data
            #####

            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            logger.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(FLAGS.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)
    

    # SyncBatchNorm
    if FLAGS.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // FLAGS.world_size, gs, single_cls,
                                            hyp=hyp, augment=True, cache=FLAGS.cache_images, rect=FLAGS.rect, rank=RANK,
                                            workers=workers, image_weights=FLAGS.image_weights, quad=FLAGS.quad, 
                                            prefix=colorstr('train: '))
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path, imgsz, batch_size // FLAGS.world_size * 2, gs, single_cls,  # testloader
                                       hyp=hyp, cache=None if noval else FLAGS.cache_images, rect=True, rank=-1,
                                       workers=FLAGS.workers, pad=0.5, prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, names, save_dir)
            
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = FLAGS.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=FLAGS.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional, single-GPU only)
        if FLAGS.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        logger.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if FLAGS.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                if FLAGS.visualize:
                    pred = model(imgs, visualize=save_dir)  # forward
                else:
                    pred = model(imgs, visualize=False)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= FLAGS.world_size  # gradient averaged between devices in DDP mode
                if FLAGS.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, True, FLAGS.sync_bn)

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # DDP process 0 or single-GPU
        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(data_dict,
                                                 batch_size=batch_size // FLAGS.world_size * 2,
                                                 imgsz=imgsz,
                                                 model=ema.ema,
                                                 single_cls=single_cls,
                                                 dataloader=val_loader,
                                                 save_dir=save_dir,
                                                 save_json=is_coco and final_epoch,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=True, ### and final_epoch,
                                                 callbacks=callbacks,
                                                 compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        logger.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        if is_coco:  # COCO dataset
            for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                results, _, _ = val.run(data_dict,
                                        batch_size=batch_size // FLAGS.world_size * 2,
                                        imgsz=imgsz,
                                        model=attempt_load(m, device).half(),
                                        iou_thres=0.7,  # NMS IoU threshold for best pycocotools results
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        save_json=True,
                                        plots=False)
        # Strip optimizers
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        callbacks.run('on_train_end', last, best, True, epoch)
        logger.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def main(argv, callbacks=Callbacks()):

    # Set DDP variables
    FLAGS.local_rank = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
    FLAGS.global_rank = int(os.getenv('RANK', -1))
    FLAGS.world_size = int(os.getenv('WORLD_SIZE', 1))
    
    set_logging(FLAGS.global_rank)

    # Resume
    if FLAGS.resume and not check_wandb_resume(FLAGS):  # resume an interrupted run
        ckpt = FLAGS.resume if isinstance(FLAGS.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        logger.info('Resuming training from %s' % ckpt)
        FLAGS.save_dir = Path(ckpt).parent.parent

    else:
        FLAGS.data, FLAGS.cfg, FLAGS.hyp = check_file(FLAGS.data), check_file(FLAGS.cfg), check_file(FLAGS.hyp)  # check files
        assert len(FLAGS.cfg) or len(FLAGS.weights), 'either --cfg or --weights must be specified'
        FLAGS.save_dir = str(increment_path(Path(FLAGS.project) / FLAGS.name, exist_ok=FLAGS.exist_ok))

    # DDP mode
    device = select_device(FLAGS.device, batch_size=FLAGS.batch_size)
    if FLAGS.local_rank != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > FLAGS.local_rank, 'insufficient CUDA devices for DDP command'
        assert FLAGS.batch_size % FLAGS.world_size == 0, '--batch-size must be multiple of CUDA device count'
        assert not FLAGS.image_weights, '--image-weights argument is not compatible with DDP training'
        torch.cuda.set_device(FLAGS.local_rank)
        device = torch.device('cuda', FLAGS.local_rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")  # distributed backend


    # Train
    train(FLAGS.hyp, device, callbacks)
    if FLAGS.world_size > 1 and FLAGS.global_rank == 0:
        logger.info('Destroying process group... ')
        dist.destroy_process_group()
        

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='objectbox.pt')
    opt = FLAGS
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__": 
    app.run(main)