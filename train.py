from src.utils import gmkdir
from src.datasets import SynthDataset, SynthCollator
from src.model import CRNN
from src.loss import CustomCTCLoss
from src.learner import Learner
from torch.utils.data import random_split
import os
import torch

if __name__ == "__main__":
    alphabet = "0123456789"
    args = {
        "name": "exp1",
        "path": "data",
        "imgdir": "images",
        "imgH": 32,
        "nChannels": 1,
        "nHidden": 256,
        "nClasses": len(alphabet) + 1,
        "lr": 0.0001,
        "epochs": 10,
        "batch_size": 32,
        "save_dir": "checkpoints",
        "log_dir": "logs",
        "resume": True,
        "cuda": False,
        "schedule": False,
    }

    data = SynthDataset(args)
    args["collate_fn"] = SynthCollator()

    num_train_data = int(0.8 * len(data))
    num_val_data = len(data) - num_train_data

    args["data_train"], args["data_val"] = random_split(
        data, (num_train_data, num_val_data)
    )
    assert num_train_data == len(args["data_train"])
    assert num_val_data == len(args["data_val"])
    print(f"{num_train_data=} {num_val_data=}")

    args["alphabet"] = alphabet
    model = CRNN(args)

    args["criterion"] = CustomCTCLoss()
    savepath = os.path.join(args["save_dir"], args["name"])
    gmkdir(savepath)
    gmkdir(args["log_dir"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    learner = Learner(model, optimizer, savepath=savepath, resume=args['resume'])
    learner.fit(args)
