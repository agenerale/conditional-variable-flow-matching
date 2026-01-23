import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import matplotlib.pyplot as plt

import models.cnf as cnf
import models.data_new as data_utils
from models.ema_callback import EMACallback
from models.utils import generate_traj, plot_traj, eval_mae

def get_args():
    parser = argparse.ArgumentParser(description="Conditional Variational Flow Matching")
    # Config
    parser.add_argument("--train", action='store_true', help="Run in training mode")
    parser.add_argument("--load", action='store_false', help="Load pretrained model from checkpoint")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs per node")
    parser.add_argument("--n_epoch", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--lr_init", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--lr_end", default=1e-8, type=float, help="Final learning rate")
    parser.add_argument("--clip", default=1, type=int, help="Gradient clipping value")
    parser.add_argument("--wdecay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--batch_size", default=5000, type=int, help="Minibatch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Dataloader workers")
    # Paths
    parser.add_argument("--ckpt", default="./logs/model.ckpt", type=str, help="Checkpoint path")

    parser.add_argument("--data_path", default='./data/memphis_dataset_seg_scores.h5', type=str, help="Path to dataset")
    # Architecture
    parser.add_argument("--drift_layers", default=3, type=int, help="Drift model layers")
    parser.add_argument("--score_layers", default=3, type=int, help="Score model layers")
    parser.add_argument("--drift_width", default=512, type=int, help="Drift model width")
    parser.add_argument("--score_width", default=512, type=int, help="Score model width")
    parser.add_argument("--cond_emb", default=64, type=int, help="Conditioning embedding dim")
    parser.add_argument("--n_attn", default=8, type=int, help="Attention heads")
    # Hyperparameters
    parser.add_argument("--ndim", default=5, type=int, help="Flow dimensions")
    parser.add_argument("--cdim", default=3, type=int, help="Conditioning dimensions")
    parser.add_argument("--sigma", default=1e-1, type=float, help="State noise")
    parser.add_argument("--sigmay", default=1e-2, type=float, help="Conditioning noise")
    parser.add_argument("--oty", default=100, type=float, help="Conditioning cost weight")
    parser.add_argument("--sigma_fac", default=1.0, type=float, help="Sigma factor for PC1")
    parser.add_argument("--scaling", action='store_false', help="Disable gradient standardization")
    parser.add_argument("--cutoff", default=0, type=int, help="Points to cut from start")
    
    return parser.parse_args()

def train_model(args, model, data_module):
    print(f'Starting training for {args.n_epoch} epochs...')
    
    trainer = pl.Trainer(
        devices=args.gpus,
        num_nodes=args.nodes,
        accelerator="auto",
        gradient_clip_val=args.clip,
        max_epochs=args.n_epoch,
        default_root_dir="logs/",
        callbacks=[EMACallback(), RichProgressBar()]
    )

    trainer.fit(model, data_module)

def run_evaluation(args, model, data_module, device):
    print("Running evaluation...")
    model.eval()

    n_sample = 128
    n_eval = 5 
    start_idx = 0 
    
    data_test = data_module.test_xo.to(device)
    
    print(f'Generating trajectories for {n_eval} test samples starting at index {start_idx}...')

    traj_test, _ = generate_traj(
        model, 
        args, 
        n_eval, 
        start_idx, 
        n_sample, 
        data_test, 
        device=device
    )

    os.makedirs("./imgs/traj", exist_ok=True)
    
    fname = f'./imgs/traj/test_trajectories_idx_{start_idx}.png'
    print(f"Saving plot to {fname}...")
    
    plot_traj(
        fname, 
        args, 
        traj_test, 
        n_eval, 
        start_idx, 
        n_sample, 
        data_test, 
        device=device
    )

    mae_test, traj_test_mean = eval_mae(model,
                                        args,
                                        start_idx,
                                        traj_test,
                                        data_test,
                                        device=device)
    print(f'MAE Test: {torch.mean(mae_test)}')   
    
    print("Evaluation Complete.")

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs("./imgs/traj", exist_ok=True)

    print('Preparing Data Module...')
    data_module = data_utils.SpinodalDataModule(args.data_path, args, device)
    data_module.prepare_data()

    print('Initializing Models...')   
    model = cnf.CombinedModel(args).to(device)

    if args.load and args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        try:
            state_dict = torch.load(args.ckpt, weights_only=False, map_location=device)
            model.load_state_dict(state_dict['state_dict'])
        except FileNotFoundError:
            print("Checkpoint not found.")

    if args.train:
        train_model(args, model, data_module)
    else:
        run_evaluation(args, model, data_module, device)

if __name__ == "__main__":
    main()