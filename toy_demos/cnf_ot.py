import math
import time
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
import matplotlib.pyplot as plt

def eight_normal_sample(n, dim, scale=1, var=1):
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
    return data

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
    
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    colors = cmap(norm(y)) 
    
    plt.figure(figsize=(8, 8))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black",zorder=5)
    for i in range(n):
        plt.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], alpha=0.1,zorder=0)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="maroon",zorder=10)
    
    plt.legend([r"$p_0$", r"$x_{t} \vert x_{0}$", r"$p_1$"])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    plt.savefig('flow_cont.png')
    
def sample_moons(n, theta):
    theta = np.radians(theta)
    rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]],dtype=torch.float32)
    
    x0, y0 = generate_moons(n, noise=0.2)
    x0 = 3*(x0 - 0.5)
    x0 = torch.matmul(x0, rot_mat)
    return x0, y0


device = torch.device("cuda")    

sigma = 0.1
sigmay = 0.01
oty = 100
dim = 2
cdim = 1
batch_size = 512

model = MLP(dim=dim, cdim=cdim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

a, b = pot.unif(batch_size), pot.unif(batch_size)
for k in range(10000):
    optimizer.zero_grad()
    t = torch.randn(batch_size, 1).to(device)
    t = torch.nn.Sigmoid()(t)
    x0, y0 = sample_moons(batch_size, 0)
    x1, y1 = sample_moons(batch_size, 270)
    y0 = torch.linalg.norm(x0,dim=1)*(-1*y0) + torch.linalg.norm(x0,dim=1)*(1 - y0)
    y1 = y0
    y0 = y0.unsqueeze(-1)
    y1 = y1.unsqueeze(-1)
    
    ind0 = torch.randperm(x0.size()[0])
    ind1 = torch.randperm(x1.size()[0])
    x0, y0 = x0[ind0].to(device), y0[ind0].to(device)
    x1, y1 = x1[ind1].to(device), y1[ind1].to(device)  
    
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
    y = muy_t + sigmay * torch.randn_like(y0).to(device)
    
    ut = x1 - x0
    vt = model(x, y, t)  

    yerr = torch.exp(-0.5*((y1-y0)/sigmay)**2)
    
    loss = torch.mean(yerr*(vt - ut) ** 2) 
    loss.backward()
    optimizer.step()
    
    if (k + 1) % 100 == 0:
        print(f"{k+1}: loss {loss.item():0.3f}")
    
    if (k + 1) % 2000 == 0:  
        x0, y0 = sample_moons(2048, 0)
        x0, y0 = x0.to(device), y0.to(device)
        y0 = torch.linalg.norm(x0,dim=1)*(-1*y0) + torch.linalg.norm(x0,dim=1)*(1 - y0)
        node = NeuralODE(
            torchdyn_wrapper(model, y0.unsqueeze(-1)), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        ).to(device)
        
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 100),
            )
            plot_trajectories(traj.detach().cpu().numpy(), y0.detach().cpu().numpy())



