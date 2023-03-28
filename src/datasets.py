import os
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import random_split
from functools import partial
import math
import random


class SynthCollator(object):
    def __call__(self, batch):
        width = [item["img"].shape[2] for item in batch]
        indexes = [item["idx"] for item in batch]
        imgs = torch.ones(
            [
                len(batch),
                batch[0]["img"].shape[0],
                batch[0]["img"].shape[1],
                max(width),
            ],
            dtype=torch.float32,
        )
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0 : item["img"].shape[2]] = item["img"]
            except:
                print(imgs.shape)
        item = {"img": imgs, "idx": indexes}
        if "label" in batch[0].keys():
            labels = [item["label"] for item in batch]
            item["label"] = labels
        return item


class SynthDataset(Dataset):
    def __init__(self, opt, is_eval=False):
        super(SynthDataset, self).__init__()
        self.path = os.path.join(opt["path"], opt["imgdir"])
        self.images = os.listdir(self.path)
        self.nSamples = len(self.images)
        f = lambda x: os.path.join(self.path, x)
        self.imagepaths = list(map(f, self.images))
        eval_transform_list = [
            transforms.Lambda(partial(self.handle_height, opt["imgH"])),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        """
        train_transform_list = [
            transforms.Lambda(partial(self.handle_height, opt["imgH"])),
            transforms.RandomApply(
                [
                    transforms.Pad(
                        [random.randint(1, 10), 0, random.randint(1, 10), 0],
                        255,
                        "constant",
                    ),
                    transforms.RandomChoice(
                        [
                            transforms.RandomRotation([0, 15]),
                            transforms.RandomRotation([345, 360]),
                        ]
                    ),
                    transforms.RandomPerspective(fill=255),
                ],
                p=0.8,
            ),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        """
        train_transform_list = eval_transform_list

        self.transform = transforms.Compose(
            eval_transform_list if is_eval else train_transform_list
        )
        self.transform = transforms.Compose(eval_transform_list)
        self.collate_fn = SynthCollator()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        imagepath = self.imagepaths[index]
        imagefile = os.path.basename(imagepath)
        img = Image.open(imagepath)
        if self.transform is not None:
            img = self.transform(img)
        item = {"img": img, "idx": index}
        item["label"] = imagefile.split("_")[0]
        return item

    def handle_height(self, wanted_h, image):
        """
        Reference by: https://discuss.pytorch.org/t/dynamic-padding-based-on-input-shape/72736/2
        """
        w, h = image.size
        h_diff = h - wanted_h

        if h_diff < 0:
            top_pad = math.floor(abs(h_diff) / 2)
            bottom_pad = math.ceil(abs(h_diff) - top_pad)
            return transforms.functional.pad(
                image, [0, top_pad, 0, bottom_pad], 0, "constant"
            )
        elif h_diff > 0:
            top = abs(h_diff / 2)
            return transforms.functional.crop(image, top, 0, wanted_h, w)
        else:
            return image
