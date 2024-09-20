import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

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

class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, w=128):
        super().__init__()
        self.dim = dim
        self.cdim = cdim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + cdim + 1, w),
            #torch.nn.LayerNorm(w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            #torch.nn.LayerNorm(w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            #torch.nn.LayerNorm(w),
            torch.nn.GELU(),
            torch.nn.Linear(w, dim),
        )

    def forward(self, x, y, t):
        in_x = torch.cat([x, y, t], dim=-1)
        out = self.net(in_x)
        return out

class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]

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
    
    
    #colors = ['maroon', 'tab:blue', 'steelblue', 'peachpuff', 'lightslategrey', 'slategrey', 'lightblue', 'lightsalmon']
    
    clr_dict = {0: 'maroon',
                1: 'tab:red',
                2: 'navy',
                3: 'steelblue',
                4: 'lightsalmon',
                5: 'slategrey',
                6: 'coral',
                7: 'cadetblue'}
    
    ytraj = np.tile(y,(100,1)).flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black",zorder=5)
    #plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c=np.array(colors)[ytraj])
    
    for i in range(n):
        plt.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], alpha=0.1,zorder=0)
    
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="maroon",zorder=10)
    
    #plt.legend([r"$p_0$", r"$x_{t} \vert x_{0}$", r"$p_1$"])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    
    plt.savefig('flow_8gauss_8gauss_rot_sbot.png')
    
    
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

def sample_moons(n, theta):
    theta = np.radians(theta)
    rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]],dtype=torch.float32)
    
    x0, y0 = generate_moons(n, noise=0.2)
    x0 = 3*(x0 - 0.5)
    x0 = torch.matmul(x0, rot_mat)
    return x0, y0

N = 1024
x0, y0 = eight_normal_sample(N, 2, scale=10, var=0.2, theta=0)
x1, y1 = eight_normal_sample(N, 2, scale=5, var=0.1, theta=45)
plt.figure(figsize=(12,12))
plt.scatter(x0[:,0], x0[:,1], alpha=0.7, c=y0)
plt.scatter(x1[:,0], x1[:,1], alpha=0.7, c=y1)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show() 
plt.savefig('target_8gauss_8gauss_rot.png')

# avg_vals = np.zeros(8)
# for i in range(8):
#     y0c = y0[y0 == i]
#     x0c = x0[y0 == i]
#     x1c = x1[y1 == i]
        
#     A, B = pot.unif(x0c.shape[0]), pot.unif(x1c.shape[0])
#     M = (torch.cdist(x0c, x1c) ** 1).detach().cpu().numpy()
    
#     d_emd = pot.emd2(A, B, M)  # direct computation of OT loss
#     avg_vals[i] = d_emd
    
# ccc
device = torch.device("cuda")    

sigma = 1#0.1
sigmay = 0.02
oty = 10 #1e2 best
dim = 2
cdim = 1
batch_size = 256

model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

start = time.time()
a, b = pot.unif(batch_size), pot.unif(batch_size)
for k in range(10000):
    optimizer.zero_grad()
    t = torch.rand(batch_size, 1).to(device)
        
    x0, y0 = eight_normal_sample(batch_size, 2, scale=10, var=0.2)
    x1, y1 = eight_normal_sample(batch_size, 2, scale=5, var=0.1, theta=45)
    
    y0 = y0.unsqueeze(-1)
    y1 = y1.unsqueeze(-1)
    
    ind0 = torch.randperm(x0.size()[0])
    ind1 = torch.randperm(x1.size()[0])
    x0, y0 = x0[ind0].to(device), y0[ind0].to(device)
    x1, y1 = x1[ind1].to(device), y1[ind1].to(device)  
    
    # Resample xy0, xy1 according to transport matrix   
    M = torch.cdist(x0, x1) ** 2 + oty*torch.cdist(y0, y1) ** 2
    M = M / M.max()
    #pi = pot.emd(a, b, M.detach().cpu().numpy())
    pi = pot.sinkhorn(a, b, M.detach().cpu().numpy(), 2*sigma**2)
    # Sample random interpolations on pi
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

    x = mu_t + sigma * torch.randn(batch_size, dim).to(device)
    y = muy_t + sigmay * torch.randn(batch_size, cdim).to(device)
    
    ut = x1 - x0
    vt = model(x, y, t)  

    #yerr = sigmay/((y1-y0)**2 + 1e-8)
    yerr = torch.exp(-0.5*((y1-y0) / sigmay)**2)
    #yerr = 1

    loss = torch.mean(yerr*(vt - ut) ** 2) 

    loss.backward()
    optimizer.step()
    
    
    if (k + 1) % 10 == 0:
        print(f"{k+1}: loss {loss.item():0.3f}, xdist: {torch.mean(torch.cdist(x0, x1) ** 2):0.3f}, ydist: {oty*torch.mean(torch.cdist(y0, y1) ** 2):0.3f}")
    if (k + 1) % 1000 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end        
        
        x0, y0 = eight_normal_sample(2048, 2, scale=10, var=0.2)
        x0, y0 = x0.to(device), y0.to(device)

        node = NeuralODE(
            torchdyn_wrapper(model, y0.unsqueeze(-1)), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        ).to(device)
        
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj.detach().cpu().numpy(), y0.detach().cpu().numpy())
            
            

# Evaluate variance
batch_size_all = [16,32,64,128,256,512]
n_epoch = 100
loss_rec = np.zeros((2,5,len(batch_size_all),n_epoch))

for l in range(2):
    for m in range(5):
        c = 0
        for batch_size in batch_size_all:
            a, b = pot.unif(batch_size), pot.unif(batch_size)
            for k in range(n_epoch):
                t = torch.rand(batch_size, 1).to(device)
                    
                x0, y0 = eight_normal_sample(batch_size, 2, scale=10, var=0.2)
                x1, y1 = eight_normal_sample(batch_size, 2, scale=5, var=0.1, theta=45)
                
                #x0, y0 = eight_normal_sample(batch_size, 2, scale=7, var=0.05, theta=180)
                #x1, y1 = sample_moons(batch_size, 0)
                
                #x0, y0 = sample_moons(batch_size, 0)
                #x1, y1 = sample_moons(batch_size, 270)
                #y0 = torch.linalg.norm(x0,dim=1)*(-1*y0) + torch.linalg.norm(x0,dim=1)*(1 - y0)
                #y1 = y0
                
                y0 = y0.unsqueeze(-1)
                y1 = y1.unsqueeze(-1)
                
                ind0 = torch.randperm(x0.size()[0])
                ind1 = torch.randperm(x1.size()[0])
                x0, y0 = x0[ind0].to(device), y0[ind0].to(device)
                x1, y1 = x1[ind1].to(device), y1[ind1].to(device)  
                
                # Resample xy0, xy1 according to transport matrix   
                M = torch.cdist(x0, x1) ** 2 + oty*torch.cdist(y0, y1) ** 2
                M = M / M.max()
                #pi = pot.emd(a, b, M.detach().cpu().numpy())
                pi = pot.sinkhorn(a, b, M.detach().cpu().numpy(), 2*sigma**2)
                # Sample random interpolations on pi
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
            
                x = mu_t + sigma * torch.randn(batch_size, dim).to(device)
                y = muy_t + sigmay * torch.randn(batch_size, cdim).to(device)
                
                ut = x1 - x0
            
                #yerr = sigmay/((y1-y0)**2 + 1e-8)
                if l == 0:
                    yerr = torch.exp(-0.5*((y1-y0) / sigmay)**2)
                else:
                    yerr = 1
                
                loss = torch.mean(yerr*(ut) ** 2)
                loss_rec[l,m,c,k] = loss.item()
            
            c += 1
    
loss_var = np.var(loss_rec,axis=-1)
max_var = np.max(loss_var,axis=1)
min_var = np.min(loss_var,axis=1)
mean_var = np.mean(loss_var,axis=1)
std_var = np.std(loss_var,axis=1)

plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
font = {'family' : 'serif','weight' : 'normal','size'   : 24}
plt.rc('font', **font)

plt.figure(figsize=(6,8), constrained_layout=True)
plt.plot(np.array(batch_size_all), mean_var[0,:],label='CVFM')
plt.plot(np.array(batch_size_all), mean_var[1,:],label='COT-FM')
plt.fill_between(np.array(batch_size_all), min_var[0,:], max_var[0,:], alpha=0.2,color='tab:blue')
plt.fill_between(np.array(batch_size_all), min_var[1,:], max_var[1,:], alpha=0.2,color='tab:orange')
plt.xticks([16,128,256,512])
plt.xlabel('batch size')
plt.ylabel('Variance')
plt.legend()
plt.savefig('obj_var_discrete.png')


ccc
            


x0, y0 = eight_normal_sample(batch_size, 2, scale=10, var=0.2)
x1, y1 = eight_normal_sample(batch_size, 2, scale=5, var=0.1, theta=45)

x0, y0 = x0.to(device), y0.to(device)
x1, y1 = x1.to(device), y1.to(device)
w2c = np.zeros((2,8))

for i in range(8):
    y0c = y0[y0 == i]
    x0c = x0[y0 == i]
    x1c = x1[y1 == i]

    if x0c.shape[0] == 0 or x1c.shape[0] == 0:
        w2c[0,i] = 100
        w2c[1,i] = 100
        continue
    
    node = NeuralODE(
        torchdyn_wrapper(model, y0c.unsqueeze(-1)), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    ).to(device)
    
    with torch.no_grad():
        traj = node.trajectory(
            x0c,
            t_span=torch.linspace(0, 1, 100),
        )
        
        A, B = pot.unif(x0c.shape[0]), pot.unif(x0c.shape[0])
        M = (torch.cdist(x0c, traj[-1,...]) ** 1).detach().cpu().numpy()
        
        d_emd = pot.emd2(A, B, M)  # direct computation of OT loss
        w2c[0,i] = d_emd
        
        A, B = pot.unif(x0c.shape[0]), pot.unif(x1c.shape[0])
        M = (torch.cdist(traj[-1,...], x1c) ** 1).detach().cpu().numpy()
        
        d_emd = pot.emd2(A, B, M)  # direct computation of OT loss
        w2c[1,i] = d_emd   
