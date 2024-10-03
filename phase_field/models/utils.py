import numpy as np
import pickle as pk
import torch
import torchsde
import h5py

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import Normalize

import models.cnf as cnf

def generate_traj(model,args,n_eval,start,n_sample,traj_xo,device='cpu'):
    
    ndim = args.ndim
    cdim = args.cdim
    n_pts = traj_xo.size(1) - args.cutoff
    
    traj = torch.zeros((n_eval, n_pts, n_sample, ndim))

    for i in range(n_eval):
        x0 = traj_xo[i+start, 0, cdim:].expand(n_sample, ndim).to(device)
        y = traj_xo[i+start, 0, :cdim].expand(n_sample, cdim).to(device)

        sde = cnf.SDE(model, y, sigma=args.sigma, sigma_fac=args.sigma_fac)
        with torch.no_grad():
            sde_traj = torchsde.sdeint(
                sde,
                x0,
                ts=torch.linspace(0, n_pts, n_pts).to(device),
                dt=1,
                dt_min=1e-3,
                #adaptive=True,
            )

        traj[i, ...] = sde_traj.detach().cpu()
        print(f'Traj #: {i}') 
        
    return traj, traj_xo[start:start+n_eval, :, cdim:].detach().cpu()  
          
def plot_traj(file,args,traj,n_eval,start,n_sample,traj_xo,device='cpu'):
    
    ndim = args.ndim
    cdim = args.cdim
    n_pts = traj_xo.size(1) - args.cutoff

    traj_mean = torch.mean(traj,-2) 

    fig, axes = plt.subplots(1, ndim, figsize=(25, 5), sharex=False, constrained_layout=True)
    for i in range(ndim):
        for j in range(n_eval):
            for k in range(n_sample):
                axes.flat[i].plot(np.linspace(0, n_pts, n_pts), traj[j, :, k, i], color='tab:blue', alpha=0.025)
            axes.flat[i].plot(np.linspace(0, n_pts, n_pts), traj_xo[j+start, args.cutoff:, cdim+i].detach().cpu(), color='tab:red', alpha=1)
            axes.flat[i].plot(np.linspace(0, n_pts, n_pts), traj_mean[j, :, i].detach().cpu(), color='black',linestyle='--', alpha=1)
        
        axes.flat[i].plot(np.linspace(0, n_pts, n_pts), traj[j, :, k, i], color='tab:blue', alpha=0.2,label=r'$x_t | x_0$')
        axes.flat[i].plot(np.linspace(0, n_pts, n_pts), traj_xo[j+start, args.cutoff:, cdim+i].detach().cpu(), color='tab:red', alpha=1,label=r'$x^E_t | x^E_0$')
        axes.flat[i].plot(np.linspace(0, n_pts, n_pts), traj_mean[j, :, i].detach().cpu(), color='black',linestyle='--', alpha=1,label=r'$E[x_t | x_0]$')
        
        axes.flat[i].set_title(r'$\alpha_{'+str(i+1)+'}$')
        axes.flat[i].set_xlabel(r'$t$')
        axes.flat[i].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[-1].legend()
    #plt.savefig(file, bbox_inches='tight')
    
def eval_mae(model,args,start,traj_pred,traj_xo,device='cpu'):

    traj_pred_mean = torch.mean(traj_pred, dim=-2)
    mae = torch.mean(torch.mean(torch.abs(traj_xo[start:start+traj_pred.shape[0],args.cutoff:,args.cdim:].detach().cpu() - traj_pred_mean),1),0)
    
    return mae, traj_pred_mean    

