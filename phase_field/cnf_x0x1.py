import argparse
import torch
import pytorch_lightning as pl

import models.cnf as cnf
import models.data as data_utils
from models.ema_callback import EMACallback
from models.utils import generate_traj, plot_traj, eval_mae

parser = argparse.ArgumentParser(description="Conditional Variational Flow Matching")
# Training
parser.add_argument("--train", action='store_true', help="train (True)")
parser.add_argument("--load", action='store_false', help="load pretrained model")
parser.add_argument("--nodes", default=1, type=int,help="number of nodes")
parser.add_argument("--gpus", default=1, type=int,help="number gpus per node")
parser.add_argument("--n_epoch", default=250, type=int,help="number of epochs for training")
parser.add_argument("--lr_init", default=1e-3, type=float,help="init. learning rate")
parser.add_argument("--lr_end", default=1e-8,type=float, help="end learning rate")
parser.add_argument("--clip", default=1, type=int, help="gradient clipping")
parser.add_argument("--wdecay", default=0.01, type=float, help="weight decay")
parser.add_argument("--batch_size", default=256,type=int, help="minibatch size")
parser.add_argument("--num_workers", default=1,type=int, help="Number of workers for dataloading")
# Checkpoint
parser.add_argument("--ckpt", default="./logs/tribom25/checkpoints/epoch=249-step=195500.ckpt",type=str, help="path to checkpoint to load")
# Model architecture
parser.add_argument("--drift_layers", default=3, type=int, help="number of layers in drift model")
parser.add_argument("--score_layers", default=3, type=int, help="number of layers in score model")
parser.add_argument("--drift_width", default=512, type=int, help="layer width drift model")
parser.add_argument("--score_width", default=512, type=int, help="layer width score model")
parser.add_argument("--cond_emb", default=64, type=int, help="conditioning embedding dimension")
parser.add_argument("--n_attn", default=8, type=int, help="number of attention heads")
# Hyperparameters
parser.add_argument("--ndim", default=5, type=int, help="number of dimensions for flow")
parser.add_argument("--cdim", default=3, type=int, help="number of conditioning dimensions")
parser.add_argument("--sigma", default=1e-1, type=float, help="noise")
parser.add_argument("--sigmay", default=1e-2, type=float, help="conditioning noise")
parser.add_argument("--oty", default=10, type=float, help="conditioning cost weighting")
parser.add_argument("--sigma_fac", default=1.0, type=float, help="factor on sigma for PC 1")
parser.add_argument("--scaling", action='store_false', help="standardize gradient")
parser.add_argument("--cutoff", default=0, type=int, help="# points to remove from start")
args = parser.parse_args()


device = torch.device('cuda')

# load data
filename = './data/memphis_dataset_seg_scores.h5'
data_module = data_utils.SpinodalDataModule(filename, args, device)
data_module.prepare_data()
print('Read in DataModule')

drift_model = cnf.MLP(dim=args.ndim,
                      cdim=args.cdim,
                      edim=args.cond_emb,
                      layers=args.drift_layers,
                      w=args.drift_width,
                      num_heads=args.n_attn).to(device)
score_model = cnf.MLP(dim=args.ndim,
                      cdim=args.cdim,
                      edim=args.cond_emb,
                      layers=args.score_layers,
                      w=args.score_width,
                      num_heads=args.n_attn)
model = cnf.CombinedModel(drift_model, score_model, args).to(device)

if args.load:
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict['state_dict'])

if args.train:
    print(f'Epoch #: {args.n_epoch}')
    trainer = pl.Trainer(devices=args.gpus,
                         num_nodes=args.nodes,
                         accelerator="auto",
                         gradient_clip_val=args.clip,
                         max_epochs=args.n_epoch,
                         default_root_dir="logs/",
                         callbacks=[EMACallback()])

    trainer.fit(model, data_module)

else:
    model.eval()
    
    # Plot trajectories in PC space and reconstructed autocorrelations
    n_sample = 128
    n_eval = 5
    start = 0
    print(f'Num. Eval: {n_eval}')
    
    # Plot select trajectories
    fname = f'./imgs/traj/train_{start}.png'
    traj_train, _ = generate_traj(model,args,n_eval,start,n_sample,data_module.train_xo.to(device),device=device)
    plot_traj(fname,args,traj_train,n_eval,start,n_sample,data_module.train_xo.to(device),device=device)
    
    fname = f'./imgs/traj/test_{start}.png'
    traj_test, _ = generate_traj(model,args,n_eval,start,n_sample,data_module.test_xo.to(device),device=device)
    plot_traj(fname,args,traj_test,n_eval,start,n_sample,data_module.test_xo.to(device),device=device)
    
    # Evaluate trajectory error
    n_eval = 10
    traj_train, _ = generate_traj(model,args,n_eval,start,n_sample,data_module.train_xo.to(device),device=device)
    mae_train, traj_train_mean = eval_mae(model,args,start,traj_train,data_module.train_xo.to(device),device=device)
    
    traj_test, _ = generate_traj(model,args,n_eval,start,n_sample,data_module.test_xo.to(device),device=device)
    mae_test, traj_test_mean = eval_mae(model,args,start,traj_test,data_module.test_xo.to(device),device=device)

    print(f'MAE Train: {torch.mean(mae_train)}')
    print(f'MAE Test: {torch.mean(mae_test)}')     
