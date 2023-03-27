from .src.model import CRNN
from .src.utils import OCRLabelConverter
from .src.datasets import SynthDataset, SynthCollator
from functools import partial
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def detect(img: Image, checkpoint_path: str):
    assert os.path.exists(checkpoint_path)

    alphabet = "0123456789"
    args = {
        "imgH": 32,
        "nChannels": 1,
        "nHidden": 256,
        "nClasses": len(alphabet) + 1,
        "save_dir": "checkpoints",
        "alphabet": alphabet,
    }

    model = CRNN(args)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    converter = OCRLabelConverter(args["alphabet"])
    
    # transform images
    custom_transform_list = [
        transforms.Lambda(partial(handle_height, args["imgH"])),
        transforms.Lambda(partial(apply_threshold_reverse, 200)),
        transforms.Grayscale(1),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ]
    custom_transform = transforms.Compose(custom_transform_list)
    img = custom_transform(img)
    
    # collate_fn
    items = [{'img': img, 'idx':0, 'label': '__'}]
    collate_fn = SynthCollator()
    batch = collate_fn(items)

    input_, targets = batch['img'].to(device), batch['label']
    print(f'{input_.size()}')
    # images.extend(input_.squeeze().detach())
    # labels.extend(targets)
    targets, lengths = converter.encode(targets)
    logits = model(input_).transpose(1, 0)
    print(f'{logits=}')
    print(f'{lengths=}')
    logits = torch.nn.functional.log_softmax(logits, 2)
    logits = logits.contiguous().cpu()
    T, B, H = logits.size()
    print(f'{T=}')
    print(f'{B=}')
    pred_sizes = torch.LongTensor([T for i in range(B)])
    probs, pos = logits.max(2)
    pos = pos.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
    print(f'{pos.data=}')
    print(f'{pred_sizes.data=}')
    print(f'{pred_sizes=}')
    print(f'{sim_preds=}')
    # predictions.extend(sim_preds)
        
    # import sys
    # sys.exit(1)
    

    # print("In detect, model", model)
    # print('In detect, checkpoint', checkpoint)
    print("In detect, converter", converter)
    print(f'{img.shape=}')

    input_ = img.to(device)
    print(f'{input_.shape=}')
    logits = model(input_).transpose(1, 0)

def handle_height(wanted_h, image):
    """
    Reference by: https://discuss.pytorch.org/t/dynamic-padding-based-on-input-shape/72736/2
    """
    w, h = image.size
    h_diff = h - wanted_h
    
    if h_diff < 0:
        top_pad = math.floor(abs(h_diff) / 2)
        bottom_pad = math.ceil(abs(h_diff) - top_pad)
        return transforms.functional.pad(image, [0, top_pad, 0, bottom_pad], 0, 'constant')
    elif h_diff > 0:
        top = abs(h_diff / 2)
        return transforms.functional.crop(image, top, 0, wanted_h, w)
    else:
        return image
    
def apply_threshold_reverse(threshold, image):
    a = image.point(lambda p: 0 if p > threshold else 255)
    # a.show()
    import random
    b = random.randrange(0, 1000, 1)
    a.save(f'{b}.png')
    
    return image.point(lambda p: 255 if p > threshold else 0)
        