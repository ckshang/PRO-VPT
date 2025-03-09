import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision import transforms
from torchvision.transforms import functional as TF


# class ADE20KDataset(Dataset):
#     def __init__(self, root, split='training', transform=None):
#         """
#         Args:
#             root (str): Path to the ADE20K dataset directory.
#             split (str): 'training' or 'validation'.
#             transform (callable, optional): Optional transform to be applied to the images.
#         """
#         self.root = root
#         self.split = split
#         self.transform = transform
#
#         self.images_dir = os.path.join(root, 'images', split)
#         self.annotations_dir = os.path.join(root, 'annotations', split)
#
#         self.image_files = sorted(os.listdir(self.images_dir))
#         self.annotation_files = sorted(os.listdir(self.annotations_dir))
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.images_dir, self.image_files[idx])
#         anno_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
#
#         image = Image.open(img_path).convert('RGB')
#         annotation = Image.open(anno_path)  # Segmentation mask
#
#         if self.transform:
#             image = self.transform(image)
#             annotation = torch.tensor(np.array(annotation), dtype=torch.long)  # (H, W)
#
#         return {'image': image, 'label': annotation}


# class ADE20KDataset(Dataset):
#     # BASE_DIR = 'ADEChallengeData2016'
#     def __init__(self, root, split='training', transform=None, crop_size=224, encode=False):
#         # self.root = os.path.join(root, self.BASE_DIR)
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.crop_size = crop_size
#         self.encode = encode
#
#         self.images_dir = os.path.join(self.root, 'images', split)
#         self.annotations_dir = os.path.join(self.root, 'annotations', split)
#
#         self.image_files = sorted(os.listdir(self.images_dir))
#         self.annotation_files = sorted(os.listdir(self.annotations_dir))
#
#         assert len(self.image_files) == len(self.annotation_files), "Mismatch between images and annotations!"
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def _sync_transform(self, img, mask):
#         i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
#         img = TF.crop(img, i, j, h, w)
#         mask = TF.crop(mask, i, j, h, w)
#
#         if random.random() > 0.5:
#             img = TF.hflip(img)
#             mask = TF.hflip(mask)
#
#         return img, mask
#
#     def _val_sync_transform(self, img, mask):
#         img = TF.center_crop(img, (self.crop_size, self.crop_size))
#         mask = TF.center_crop(mask, (self.crop_size, self.crop_size))
#         return img, mask
#
#     def _mask_transform(self, mask):
#         mask = np.array(mask, dtype=np.int32) - 1
#         if self.encode:
#             h, w = mask.shape
#             encoded_mask = np.zeros((h, w * 2), dtype=np.int32)
#             encoded_mask[:, :w] = mask
#             encoded_mask[:, w:] = mask
#             mask = encoded_mask
#         return torch.LongTensor(mask)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.images_dir, self.image_files[idx])
#         anno_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
#
#         image = Image.open(img_path).convert('RGB')
#         mask = Image.open(anno_path)
#
#         if self.split == 'training':
#             image, mask = self._sync_transform(image, mask)
#         elif self.split == 'validation':
#             image, mask = self._val_sync_transform(image, mask)
#
#         if self.transform:
#             image = self.transform(image)
#         mask = self._mask_transform(mask)
#
#         return {'image': image, 'label': mask}


def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            # maskname = basename + '_seg' + '.png'
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def make_border(self, mask = None, crop_size=-1):
        assert(mask!=None and crop_size!=-1)
        target = Resize((crop_size // 8 ), Image.NEAREST)(mask)
        return target

    def _val_sync_transform(self, img, mask, encode_flag=True, crop_size=-1):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        if (encode_flag):
            mask = self.make_border(mask, crop_size)
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        mask = np.maximum(mask, 0)

        return img, mask

    def _sync_transform(self, img, mask, encode_flag=False, crop_size=-1):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        if(encode_flag):
            mask = self.make_border(mask, crop_size)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        mask = np.maximum(mask, 0)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


class ADE20KDataset(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    dataset_root : string
        Path to ADE20K folder.
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """

    # BASE_DIR = 'ADEChallengeData2016'
    NUM_CLASS = 150

    def __init__(self, dataset_root, split='test', mode=None, transform=None, crop_size=224, encode=False, **kwargs):
        super(ADE20KDataset, self).__init__(dataset_root, split, mode, transform, **kwargs)
        self.encode = encode
        self.crop_size = crop_size
        # dataset_root = os.path.join(dataset_root, self.BASE_DIR)
        assert os.path.exists(dataset_root), "Please setup the dataset using ../datasets/ade20k.py"
        self.images, self.masks = _get_ade20k_pairs(dataset_root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + dataset_root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), dataset_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, encode_flag=self.encode, crop_size=self.crop_size)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask, encode_flag=self.encode, crop_size=self.crop_size)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img, encode_flag=self.encode), self._mask_transform(mask, encode_flage=self.encode)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask, encode_flage=False):
        if(encode_flage):
            width, height = mask.size
            mask_ = Image.new('L', (2 * width, height), (0))
            mask_.paste(mask, (0, 0, width, height))
            mask_.paste(mask, (width, 0, 2 * width, height))
            mask = mask_
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("wall", "building, edifice", "sky", "floor, flooring", "tree",
                "ceiling", "road, route", "bed", "windowpane, window", "grass",
                "cabinet", "sidewalk, pavement",
                "person, individual, someone, somebody, mortal, soul",
                "earth, ground", "door, double door", "table", "mountain, mount",
                "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
                "chair", "car, auto, automobile, machine, motorcar",
                "water", "painting, picture", "sofa, couch, lounge", "shelf",
                "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
                "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
                "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
                "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
                "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
                "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
                "grandstand, covered stand", "path", "stairs, steps", "runway",
                "case, display case, showcase, vitrine",
                "pool table, billiard table, snooker table", "pillow",
                "screen door, screen", "stairway, staircase", "river", "bridge, span",
                "bookcase", "blind, screen", "coffee table, cocktail table",
                "toilet, can, commode, crapper, pot, potty, stool, throne",
                "flower", "book", "hill", "bench", "countertop",
                "stove, kitchen stove, range, kitchen range, cooking stove",
                "palm, palm tree", "kitchen island",
                "computer, computing machine, computing device, data processor, "
                "electronic computer, information processing system",
                "swivel chair", "boat", "bar", "arcade machine",
                "hovel, hut, hutch, shack, shanty",
                "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
                "motorcoach, omnibus, passenger vehicle",
                "towel", "light, light source", "truck, motortruck", "tower",
                "chandelier, pendant, pendent", "awning, sunshade, sunblind",
                "streetlight, street lamp", "booth, cubicle, stall, kiosk",
                "television receiver, television, television set, tv, tv set, idiot "
                "box, boob tube, telly, goggle box",
                "airplane, aeroplane, plane", "dirt track",
                "apparel, wearing apparel, dress, clothes",
                "pole", "land, ground, soil",
                "bannister, banister, balustrade, balusters, handrail",
                "escalator, moving staircase, moving stairway",
                "ottoman, pouf, pouffe, puff, hassock",
                "bottle", "buffet, counter, sideboard",
                "poster, posting, placard, notice, bill, card",
                "stage", "van", "ship", "fountain",
                "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
                "canopy", "washer, automatic washer, washing machine",
                "plaything, toy", "swimming pool, swimming bath, natatorium",
                "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
                "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
                "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
                "trade name, brand name, brand, marque", "microwave, microwave oven",
                "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
                "bicycle, bike, wheel, cycle", "lake",
                "dishwasher, dish washer, dishwashing machine",
                "screen, silver screen, projection screen",
                "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
                "traffic light, traffic signal, stoplight", "tray",
                "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
                "dustbin, trash barrel, trash bin",
                "fan", "pier, wharf, wharfage, dock", "crt screen",
                "plate", "monitor, monitoring device", "bulletin board, notice board",
                "shower", "radiator", "glass, drinking glass", "clock", "flag")
