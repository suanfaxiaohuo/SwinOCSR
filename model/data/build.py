# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp
from datasets import *
from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from PIL import Image





def SMILES_build_loader_500wan(config,dir):
    # config.defrost()
    # dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    # config.freeze()
    data_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.RandomErasing(p=0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data_transform_val = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        #transforms.RandomRotation(15),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.RandomErasing(p=0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset_train = CaptionDataset_500wan(dir, 'train', transform=data_transform)

    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val = CaptionDataset_500wan(dir, 'val', transform=data_transform_val)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
    dataset_test = CaptionDataset_500wan(dir, 'test', transform=data_transform_val)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=256,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return  data_loader_train, data_loader_val, data_loader_test, mixup_fn



def SMILES_build_loader_500wan_384(config,dir):
    # config.defrost()
    # dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    # config.freeze()
    data_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((384, 384)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.RandomErasing(p=0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data_transform_val = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((384, 384)),
        #transforms.RandomRotation(15),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.RandomErasing(p=0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset_train = CaptionDataset_500wan(dir, 'train', transform=data_transform)

    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val = CaptionDataset_500wan(dir, 'val', transform=data_transform_val)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
    dataset_test = CaptionDataset_500wan(dir, 'test', transform=data_transform_val)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=128,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return  data_loader_train, data_loader_val, data_loader_test, mixup_fn





def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
