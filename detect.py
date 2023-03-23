from .src.model import CRNN
from .src.utils import OCRLabelConverter
from .src.datasets import SynthDataset, SynthCollator
from PIL import Image
import os
import torch


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
    
    print(f'{img.size=}')

    
    dataset = SynthDataset({"path": ".", "imgdir": "."})
    assert dataset.transform is not None
    img = dataset.transform(img)
    items = [{'img': img, 'idx':0, 'label': '_'}]
    collate_fn = SynthCollator()
    batch = collate_fn(items)

    input_, targets = batch['img'].to(device), batch['label']
    # images.extend(input_.squeeze().detach())
    # labels.extend(targets)
    targets, lengths = converter.encode(targets)
    logits = model(input_).transpose(1, 0)
    print(f'{logits=}')
    # logits = torch.nn.functional.log_softmax(logits, 2)
    # logits = logits.contiguous().cpu()
    # T, B, H = logits.size()
    # pred_sizes = torch.LongTensor([T for i in range(B)])
    # probs, pos = logits.max(2)
    # pos = pos.transpose(1, 0).contiguous().view(-1)
    # sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
    # predictions.extend(sim_preds)
        
    import sys
    sys.exit(1)
    

    # print("In detect, model", model)
    # print('In detect, checkpoint', checkpoint)
    print("In detect, converter", converter)
    print(f'{img.shape=}')

    input_ = img.to(device)
    print(f'{input_.shape=}')
    logits = model(input_).transpose(1, 0)
