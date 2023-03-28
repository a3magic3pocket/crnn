from src.utils import OCRLabelConverter, Eval
from src.datasets import SynthDataset, SynthCollator
from src.model import CRNN
from torchvision.utils import make_grid
from tqdm import tqdm
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_accuracy(args):
    loader = torch.utils.data.DataLoader(args['data'],
                batch_size=args['batch_size'],
                collate_fn=args['collate_fn'])
    model = args['model']
    model.eval()
    converter = OCRLabelConverter(args['alphabet'])
    evaluator = Eval()
    labels, predictions, images = [], [], []
    for iteration, batch in enumerate(tqdm(loader)):
        input_, targets = batch['img'].to(device), batch['label']
        images.extend(input_.squeeze().detach())
        labels.extend(targets)
        targets, lengths = converter.encode(targets)
        logits = model(input_).transpose(1, 0)
        logits = torch.nn.functional.log_softmax(logits, 2)
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
        predictions.extend(sim_preds)
        
#     make_grid(images[:10], nrow=2)
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        img = images[i]
        img = (img - img.min())/(img.max() - img.min())
        img = np.array(img * 255.0, dtype=np.uint8)
        fig.add_subplot(rows, columns, i)
        plt.title(predictions[i])
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    ca = np.mean((list(map(evaluator.char_accuracy, list(zip(predictions, labels))))))
    wa = np.mean((list(map(evaluator.word_accuracy_line, list(zip(predictions, labels))))))
    return ca, wa

if __name__ == "__main__":
    alphabet = "0123456789"
    args = {
        "name": "exp1",
        "path": "data",
        "imgdir": "venus_test_one",
        "imgH": 32,
        "nChannels": 1,
        "nHidden": 256,
        "nClasses": len(alphabet) + 1,
        "batch_size": 32,
        "save_dir": "checkpoints",
        'ckpt_name': 'best.ckpt',
        "alphabet": alphabet,
    }
    
    args['data'] = SynthDataset(args, is_eval=True)
    args["collate_fn"] = SynthCollator()
    resume_file = os.path.join(args['save_dir'], args['name'], args['ckpt_name'])
    model = CRNN(args)
    if os.path.isfile(resume_file):
        print('Loading model %s'%resume_file)
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint['state_dict'])
        args['model'] = model
        ca, wa = get_accuracy(args)
        print("Character Accuracy: %.2f\nWord Accuracy: %.2f"%(ca, wa))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))
        print('Exiting')