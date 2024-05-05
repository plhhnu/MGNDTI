from models import MGNDTI
import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="MGNDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='sample')
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    output_path = os.path.join(cfg.RESULT.OUTPUT_DIR, args.data)
    cfg.RESULT.OUTPUT_DIR = output_path
    mkdir(output_path)

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    dataFolder = f'../datasets/{args.data}/'

    """load data"""
    train_set = pd.read_csv(dataFolder + "train.csv")
    val_set = pd.read_csv(dataFolder + "val.csv")
    test_set = pd.read_csv(dataFolder + "test.csv")
    print(f"train_set: {len(train_set)}")
    print(f"val_set: {len(val_set)}")
    print(f"test_set: {len(test_set)}")

    set_seed(cfg.SOLVER.SEED)
    train_dataset = DTIDataset(train_set.index.values, train_set)
    val_dataset = DTIDataset(val_set.index.values, val_set)
    test_dataset = DTIDataset(test_set.index.values, test_set)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                                                                   'drop_last': True, 'collate_fn': graph_collate_func}
    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = MGNDTI(**cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, **cfg)

    result = trainer.train()

    with open(os.path.join(output_path, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    return result

if __name__ == '__main__':
    print(f"start: {datetime.now()}")
    start_time = time.time()
    """ train """
    result = main()
    """"""
    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60
    seconds = total_time_seconds % 60
    print("Total running time of the model: {} hours {} minutes {} seconds".format(int(hours), int(minutes),
                                                                              int(seconds)))
    print(f"end: {datetime.now()}")