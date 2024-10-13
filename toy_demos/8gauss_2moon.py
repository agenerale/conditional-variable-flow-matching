import math
import time
import numpy as np
import ot as pot
import torch
import torchdyn
import torchsde
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
import matplotlib.pyplot as plt

font = {'size' : 20}
plt.rc('font', **font)

def eight_normal_sample(n, dim, scale=1, var=1, theta=0):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    
    theta = np.radians(theta)
    rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]],dtype=torch.float32)
    data = torch.matmul(data.float(), rot_mat)
   
    return data, multi.float()

def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()

class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, w=128):
        super().__init__()
        self.dim = dim
        self.cdim = cdim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + cdim + 1, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, dim),
        )

    def forward(self, x, y, t):
        in_x = torch.cat([x, y, t], dim=-1)
        out = self.net(in_x)
        return out

class torchdyn_wrapper(torch.nn.Module):
    def __init__(self, model, y):
        super().__init__()
        self.model = model
        self.y = y
        
    def forward(self, t, x, args=None):  
        return self.model(x, self.y, t.repeat(x.shape[0])[:, None]) 

def plot_trajectories(traj, y):   
    n = 2000
    y = y[:n].astype('int')
    
    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    colors = cmap(norm(y)) 
    
    plt.figure(figsize=(8, 8))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black",zorder=5)
    for i in range(n):
        plt.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], alpha=0.1,zorder=0)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="maroon",zorder=10)
    #plt.legend([r"$p_0$", r"$x_{t} \vert x_{0}$", r"$p_1$"])
    plt.axis('off')
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.show()
    

def sample_moons(n, theta):
    theta = np.radians(theta)
    rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]],dtype=torch.float32)
    
    x0, y0 = generate_moons(n, noise=0.2)
    x0 = 3*(x0 - 0.5)
    x0 = torch.matmul(x0, rot_mat)
    
    ind0 = np.where(y0 == 0)[0]
    ind1 = np.where(y0 == 1)[0]
    cntr0 = np.array([(x0[ind0,0].max() - x0[ind0,0].min())/2 + x0[ind0,0].min(),
                      x0[ind0,1].min()])
    cntr1 = np.array([(x0[ind1,0].max() - x0[ind1,0].min())/2 + x0[ind1,0].min(),
                      x0[ind1,1].max()])
    ang0 = np.arctan2(x0[ind0,1] - cntr0[1],x0[ind0,0] - cntr0[0])
    ang1 = -np.arctan2(x0[ind1,1] - cntr1[1],x0[ind1,0] - cntr1[0])
    
    incr = np.pi/4
    
    for i in range(4):
        y0[ind0[(ang0 >= i*incr) & (ang0 <= (i+1)*incr)]] = i
        y0[ind1[(ang1 >= i*incr) & (ang1 <= (i+1)*incr)]] = 7 - i       
    
    return x0, y0.float()

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift_model, score_model, cond, sigma=1.0, sigma_fac=1.0):
        super().__init__()
        self.drift_model = drift_model
        self.score_model = score_model
        self.sigma = sigma
        self.sigma_fac = sigma_fac
        self.cond = cond

    # Drift
    def f(self, t, y):
        t = t.repeat(y.shape[0])[:, None]
        vt = self.drift_model(y,self.cond,t)
        st = self.score_model(y,self.cond,t)
        return vt + st

    # Diffusion
    def g(self, t, y):
        out = torch.ones_like(y) * self.sigma 
        return out


def train_ode(model, params, method='cvfm'):

    ''' train continuous 2 moons - 2 moons example (ODE) 
        methods = ['cfvm','cot-fm','cfm'] '''
        
    sigma, sigmay, oty, batch_size = params

    a, b = pot.unif(batch_size), pot.unif(batch_size)
    for k in range(10000):
        optimizer.zero_grad()
        t = torch.randn(batch_size, 1).to(device)
        t = torch.nn.Sigmoid()(t)

        x0, y0 = eight_normal_sample(batch_size, 2, scale=7, var=0.05, theta=180)
        x1, y1 = sample_moons(batch_size, 0)
            
        y0 = y0.unsqueeze(-1)
        y1 = y1.unsqueeze(-1)
        
        ind0 = torch.randperm(x0.size()[0])
        ind1 = torch.randperm(x1.size()[0])
        x0, y0 = x0[ind0].to(device), y0[ind0].to(device)
        x1, y1 = x1[ind1].to(device), y1[ind1].to(device)  
        
        if method == 'cvfm' or method =='cot-fm':
            # Resample xy0, xy1 according to transport matrix
            M = torch.cdist(x0, x1) ** 2 + oty*torch.cdist(y0, y1) ** 2
            M = M / M.max()
            pi = pot.emd(a, b, M.detach().cpu().numpy())
            p = pi.flatten()
            p = p / p.sum()
            choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=False)
            i, j = np.divmod(choices, pi.shape[1])   
            x0 = x0[i]
            x1 = x1[j]
            y0 = y0[i] 
            y1 = y1[j]
        
        # calculate regression loss
        mu_t = x0 * (1 - t) + x1 * t
        muy_t = y0 * (1 - t) + y1 * t
        
        x = mu_t + sigma * torch.randn_like(x0).to(device)
        if method == 'cfm':
            y = muy_t
        else:
            y = muy_t + sigmay * torch.randn_like(y0).to(device)
        
        ut = x1 - x0
        vt = model(x, y, t)  
    
        if method == 'cvfm':
            yerr = torch.exp(-0.5*((y1-y0)/sigmay)**2)
        else:
            yerr = 1
        
        loss = torch.mean(yerr*(vt - ut) ** 2) 
        loss.backward()
        optimizer.step()
        
        if (k + 1) % 100 == 0:
            print(f"{k+1}: loss {loss.item():0.3f}")
            
    

    x0, y0 = eight_normal_sample(2048, 2, scale=7, var=0.05, theta=180)
    x0, y0 = x0.to(device), y0.to(device)
    
    node = NeuralODE(
        torchdyn_wrapper(model, y0.unsqueeze(-1)), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    ).to(device)
    
    with torch.no_grad():
        traj = node.trajectory(
            x0,
            t_span=torch.linspace(0, 1, 100),
        )
        
    return traj.detach().cpu().numpy(), y0.detach().cpu().numpy()

def train_sde(drift_model, score_model, params, method='cvsfm'):

    ''' train continuous 2 moons - 2 moons example (ODE) 
        methods = ['cvsfm','cot-sfm','csfm'] '''
        
    sigma, sigmay, oty, batch_size = params

    a, b = pot.unif(batch_size), pot.unif(batch_size)
    for k in range(10000):
        optimizer.zero_grad()
        t = torch.randn(batch_size, 1).to(device)
        t = torch.nn.Sigmoid()(t)
        x0, y0 = eight_normal_sample(batch_size, 2, scale=7, var=0.05, theta=180)
        x1, y1 = sample_moons(batch_size, 0)
        
        y0 = y0.unsqueeze(-1)
        y1 = y1.unsqueeze(-1)
        
        ind0 = torch.randperm(x0.size()[0])
        ind1 = torch.randperm(x1.size()[0])
        x0, y0 = x0[ind0].to(device), y0[ind0].to(device)
        x1, y1 = x1[ind1].to(device), y1[ind1].to(device)  
        
        if method == 'cvfm' or method =='cot-fm':
            # Resample xy0, xy1 according to transport matrix
            M = torch.cdist(x0, x1) ** 2 + oty*torch.cdist(y0, y1) ** 2
            M = M / M.max()
            pi = pot.emd(a, b, M.detach().cpu().numpy())
            p = pi.flatten()
            p = p / p.sum()
            choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=False)
            i, j = np.divmod(choices, pi.shape[1])   
            x0 = x0[i]
            x1 = x1[j]
            y0 = y0[i] 
            y1 = y1[j]
        
        # Construct estimate for velocity with conditional OT
        sigma_t = sigma * torch.sqrt(t * (1 - t))
        sigmay_t = sigmay * torch.sqrt(t * (1 - t))
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
    
        eps = torch.randn_like(x0).to(device)
    
        mu_t = x0 * (1 - t) + x1 * t
        xt = mu_t + sigma_t * eps
    
        epsy = torch.randn_like(y0).to(device)
        muy_t = y0 * (1 - t) + y1 * t
        
        if method == 'csfm':
            yt = muy_t
        else:
            yt = muy_t + sigmay_t * epsy
        
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + (x1 - x0)
        lambda_t = 2 * sigma_t / (sigma**2 + 1e-8)  
    
        vt = drift_model(xt, yt, t)
        st = score_model(xt, yt, t)
            
        
        if method == 'cvsfm':
            yerr = torch.exp(-0.5*((y1-y0)/sigmay)**2)
        else:
            yerr = 1
            
        drift_loss = torch.mean(yerr*(vt - ut)**2) 
        score_loss = torch.mean(yerr*(lambda_t[:, None] * st + eps) ** 2)
        
        loss = torch.sum(torch.stack((score_loss,drift_loss)))
    
        loss.backward()
        optimizer.step()
        
        if (k + 1) % 100 == 0:
            print(f"{k+1}: loss {loss.item():0.3f}")
            
    
    x0, y0 = eight_normal_sample(2048, 2, scale=7, var=0.05, theta=180)
    x0, y0 = x0.to(device), y0.to(device)
    
    sde = SDE(drift_model, score_model, y0.unsqueeze(-1), sigma=sigma)
    with torch.no_grad():
        traj = torchsde.sdeint(
            sde,
            x0,
            ts=torch.linspace(0, 1, 100).to(device),
            dt=1e-2,
            dt_min=1e-4,
            adaptive=True,
            )
            
        
    return traj.detach().cpu().numpy(), y0.detach().cpu().numpy()
        
        
device = torch.device("cuda")    

sigma = 0.1
sigmay = 0.01
oty = 100
dim = 2
cdim = 1
batch_size = 256

params = (sigma, sigmay, oty, batch_size)


model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
traj_cvfm, y0_cvfm = train_ode(model, params, method='cvfm')

model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
traj_cot_fm, y0_cot_fm = train_ode(model, params, method='cot-fm')

model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
traj_cfm, y0_cfm = train_ode(model, params, method='cfm')

sigma = 0.5
sigmay = 0.01
oty = 100
dim = 2
cdim = 1
batch_size = 256

params = (sigma, sigmay, oty, batch_size)

drift_model = MLP(dim=dim, cdim=cdim).to(device)
score_model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.Adam(list(drift_model.parameters()) + list(score_model.parameters()), lr=1e-3)
traj_cvsfm, y0_cvsfm = train_sde(drift_model, score_model, params, method='cvsfm')

drift_model = MLP(dim=dim, cdim=cdim).to(device)
score_model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.Adam(list(drift_model.parameters()) + list(score_model.parameters()), lr=1e-3)
traj_cot_sfm, y0_cot_sfm = train_sde(drift_model, score_model, params, method='cot-sfm')

drift_model = MLP(dim=dim, cdim=cdim).to(device)
score_model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.Adam(list(drift_model.parameters()) + list(score_model.parameters()), lr=1e-3)
traj_csfm, y0_csfm = train_sde(drift_model, score_model, params, method='csfm')


def plot_trajectories_subplot(traj, y, title, j, ax):
    n = 2000
    y = y[:n].astype('int')
    
    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    colors = cmap(norm(y)) 
    
    ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black",zorder=5)
    for i in range(n):
        ax.plot(traj[:j, i, 0], traj[:j, i, 1], color=colors[i], alpha=0.1,zorder=0)
    ax.scatter(traj[j, :n, 0], traj[j, :n, 1], s=4, alpha=1, c="maroon",zorder=10)
    #ax.legend([r"$p_0$", r"$x_{t} \vert x_{0}$", r"$p_1$"])
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    ax.axis('off')
    ax.set_title(title)
    #ax.show()
        
def plot_hist_subplot(traj, y, title, j, ax):
    n = 2000
    y = y[:n].astype('int')
        
    ax.hist2d(*traj[j, :n, :].T, bins=256, range=((-10, 10), (-10, 10)))
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    #ax.set_title(title)
    ax.axis('off')


import matplotlib.animation as animation

fig, axes = plt.subplots(2,6, figsize=(25,10), constrained_layout=True)

def animate(i):
    # ODE
    axes[0,0].cla()
    plot_trajectories_subplot(traj_cvfm, y0_cvfm, 'CVFM', i, axes[0,0])
    
    axes[0,1].cla()
    plot_trajectories_subplot(traj_cot_fm, y0_cot_fm, 'COT-FM', i, axes[0,1])

    axes[0,2].cla()
    plot_trajectories_subplot(traj_cfm, y0_cfm, 'CFM', i, axes[0,2])
    
    # SDE
    axes[0,3].cla()
    plot_trajectories_subplot(traj_cvsfm, y0_cvsfm, 'CVSFM', i, axes[0,3])
    
    axes[0,4].cla()
    plot_trajectories_subplot(traj_cot_sfm, y0_cot_sfm, 'COT-SFM', i, axes[0,4])

    axes[0,5].cla()
    plot_trajectories_subplot(traj_csfm, y0_csfm, 'CSFM', i, axes[0,5])
    
    
    # ODE
    axes[1,0].cla()
    plot_hist_subplot(traj_cvfm, y0_cvfm, 'CVFM', i, axes[1,0])
    
    axes[1,1].cla()
    plot_hist_subplot(traj_cot_fm, y0_cot_fm, 'COT-FM', i, axes[1,1])

    axes[1,2].cla()
    plot_hist_subplot(traj_cfm, y0_cfm, 'CFM', i, axes[1,2])
    
    # SDE
    axes[1,3].cla()
    plot_hist_subplot(traj_cvsfm, y0_cvsfm, 'CVSFM', i, axes[1,3])
    
    axes[1,4].cla()
    plot_hist_subplot(traj_cot_sfm, y0_cot_sfm, 'COT-SFM', i, axes[1,4])

    axes[1,5].cla()
    plot_hist_subplot(traj_csfm, y0_csfm, 'CSFM', i, axes[1,5])
    
    for ax in axes.flat:
        ax.set_aspect('equal')
    
ani = animation.FuncAnimation(fig, animate, frames=len(traj_cvfm), interval=5, repeat=False)
plt.show()
ani.save('../imgs/8gauss_to_2moons.gif', writer='imagemagick', fps=15)
