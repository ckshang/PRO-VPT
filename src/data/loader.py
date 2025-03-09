#!/usr/bin/env python3

"""Data loader."""
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from PIL import Image
from torchvision import transforms
from pathlib import Path

from ..utils import logging
from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
)
from .datasets.ade20k_dataset import ADE20KDataset
from .datasets.coco import CocoDetection
from . import transforms_coco as T
from .util import misc as U
# from torchvision.datasets import CocoDetection


logger = logging.get_logger("visual_prompt")
_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
}

_VTAB_DATASET_CATALOG = [
    'vtab-caltech101',
    'vtab-cifar(num_classes=100)',
    'vtab-dtd',
    'vtab-oxford_flowers102',
    'vtab-oxford_iiit_pet',
    'vtab-patch_camelyon',
    'vtab-sun397',
    'vtab-svhn',
    'vtab-resisc45',
    'vtab-eurosat',
    'vtab-dmlab',
    'vtab-kitti(task="closest_vehicle_distance")',
    'vtab-smallnorb(predicted_attribute="label_azimuth")',
    'vtab-smallnorb(predicted_attribute="label_elevation")',
    'vtab-dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'vtab-dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'vtab-clevr(task="closest_object_distance")',
    'vtab-clevr(task="count_all")',
    'vtab-diabetic_retinopathy(config="btgraham-300")',
]

_VTAB_CANNOT_DOWNLOAD = [
    'vtab-patch_camelyon',
    'vtab-kitti',
    'vtab-resisc45',
    'vtab-dsprites_ori',
    'vtab-dsprites_loc',
    'vtab-cifar',
    'vtab-diabetic_retinopathy',
    'vtab-clevr_count',
    'vtab-clevr_dist',
]


def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(cfg, split)
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        dataset = _DATASET_CATALOG[dataset_name](cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def _construct_ade20k_loader(cfg, split, batch_size, shuffle, drop_last):
    root = cfg.DATA.DATAPATH + '/ADE20k'
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    if split == "train":
        dataset = ADE20KDataset(root, split='train', transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
        )
    elif split == "val":
        loader = None
    else:  # test
        dataset = ADE20KDataset(root, split='val', transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
        )

    return loader


def _construct_coco_loader(cfg, split, batch_size, shuffle, drop_last):
    root = cfg.DATA.DATAPATH + '/COCO'
    root = Path(root)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ])
        ),
        T.Resize((224, 224)),
        normalize,
    ])
    if split == "train":
        img_folder, ann_file = PATHS["train"]
        dataset = CocoDetection(img_folder, ann_file, transforms=transform)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler_train,
            collate_fn=U.collate_fn,
            num_workers=cfg.DATA.NUM_WORKERS,
        )
    elif split == "val":
        loader = None
    else:  # test
        img_folder, ann_file = PATHS["val"]
        dataset = CocoDetection(img_folder, ann_file, transforms=transform)
        sampler_val = torch.utils.data.SequentialSampler(dataset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size,
            sampler=sampler_val,
            collate_fn=U.collate_fn,
            num_workers=cfg.DATA.NUM_WORKERS,
        )
    # root = cfg.DATA.DATAPATH + '/COCO'
    # mode = 'instances'
    # PATHS = {
    #     "train": (root + "/train2017", root + "/annotations" + f'/{mode}_train2017.json'),
    #     "val": (root + "/val2017", root + "/annotations" + f'/{mode}_val2017.json'),
    # }
    # # transform = T.Compose([
    # #     T.ToTensor(),
    # # ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # if split == "train":
    #     img_folder, ann_file = PATHS["train"]
    #     dataset = CocoDetection(img_folder, ann_file, transforms=transform)
    #     loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=64,
    #         shuffle=True,
    #         num_workers=cfg.DATA.NUM_WORKERS,
    #         collate_fn=lambda x: tuple(zip(*x)),
    #     )
    # elif split == "val":
    #     loader = None
    # else:  # test
    #     img_folder, ann_file = PATHS["val"]
    #     dataset = CocoDetection(img_folder, ann_file, transforms=transform)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset)
    #     loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=64,
    #         shuffle=False,
    #         num_workers=cfg.DATA.NUM_WORKERS,
    #         collate_fn=lambda x: tuple(zip(*x))
    #     )

    return loader


def _construct_vtab_loader2(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME

    root = cfg.DATA.DATAPATH + '/' + dataset_name.split("vtab-")[-1]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    if split == "train":
        train_flist = root + "/train800.txt"
        loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=train_flist, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=drop_last,
        )
    elif split == "trainval":
        trainval_flist = root + "/train800val200.txt"
        loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=trainval_flist, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=drop_last,
        )
    elif split == "val":
        val_flist = root + "/val200.txt"
        loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=val_flist, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=drop_last,
        )
    elif split == "test":
        test_flist = root + "/test.txt"
        loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=test_flist, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=drop_last,
        )

    return loader


def construct_train_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False

    if cfg.DATA.NAME not in _VTAB_CANNOT_DOWNLOAD and cfg.DATA.NAME not in ["ade20k", "coco"]:
        return _construct_loader(
            cfg=cfg,
            split="train",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )
    elif cfg.DATA.NAME == "coco":
        return _construct_coco_loader(
            cfg=cfg,
            split="train",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )
    elif cfg.DATA.NAME == "ade20k":
        return _construct_ade20k_loader(
            cfg=cfg,
            split="train",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )
    else:
        return _construct_vtab_loader2(
            cfg=cfg,
            split="train",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )


def construct_trainval_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    if cfg.DATA.NAME not in _VTAB_CANNOT_DOWNLOAD and cfg.DATA.NAME not in ["ade20k", "coco"]:
        return _construct_loader(
            cfg=cfg,
            split="trainval",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )
    elif cfg.DATA.NAME == "coco":
        return _construct_coco_loader(
            cfg=cfg,
            split="train",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )
    elif cfg.DATA.NAME == "ade20k":
        return _construct_ade20k_loader(
            cfg=cfg,
            split="train",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )
    else:
        return _construct_vtab_loader2(
            cfg=cfg,
            split="trainval",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=drop_last,
        )


def construct_test_loader(cfg):
    """Test loader wrapper."""
    if cfg.DATA.NAME not in _VTAB_CANNOT_DOWNLOAD and cfg.DATA.NAME not in ["ade20k", "coco"]:
        return _construct_loader(
            cfg=cfg,
            split="test",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=False,
            drop_last=False,
        )
    elif cfg.DATA.NAME == "coco":
        return _construct_coco_loader(
            cfg=cfg,
            split="test",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=False,
            drop_last=False,
        )
    elif cfg.DATA.NAME == "ade20k":
        return _construct_ade20k_loader(
            cfg=cfg,
            split="test",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=False,
            drop_last=False,
        )
    else:
        return _construct_vtab_loader2(
            cfg=cfg,
            split="test",
            batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=False,
            drop_last=False,
        )


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    if cfg.DATA.NAME not in _VTAB_CANNOT_DOWNLOAD and cfg.DATA.NAME not in ["ade20k", "coco"]:
        return _construct_loader(
            cfg=cfg,
            split="val",
            batch_size=bs,
            shuffle=False,
            drop_last=False,
        )
    elif cfg.DATA.NAME == "coco":
        return _construct_coco_loader(
            cfg=cfg,
            split="val",
            batch_size=bs,
            shuffle=False,
            drop_last=False,
        )
    elif cfg.DATA.NAME == "ade20k":
        return _construct_ade20k_loader(
            cfg=cfg,
            split="val",
            batch_size=bs,
            shuffle=False,
            drop_last=False,
        )
    else:
        return _construct_vtab_loader2(
            cfg=cfg,
            split="val",
            batch_size=bs,
            shuffle=False,
            drop_last=False,
        )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        sample = {
            "image": img,
            "label": target,
        }
        return sample

    def __len__(self):
        return len(self.imlist)
