import torch
import pytorch_lightning as pl
import ot as pot
import numpy as np

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w=512):
        super(ResidualBlock, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, w),
            torch.nn.LayerNorm(w),
            torch.nn.GELU(),
            torch.nn.Linear(w, out_dim),
            )
        self.gelu = torch.nn.GELU()
        self.ln = torch.nn.LayerNorm(out_dim)
        
    def forward(self, x, yt):
        in_x = torch.cat([x, yt], -1)
        residual = x
        out = self.net(in_x)
        out += residual
        out = self.ln(out)
        out = self.gelu(out)
        return out
    
class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, edim, layers=3, w=512, w_embed=32, num_heads=8):
        super(MLP, self).__init__()
        
        self.first_layer = torch.nn.Sequential(
            torch.nn.Linear(edim + dim, w),
            torch.nn.LayerNorm(w),
            torch.nn.GELU(),
        )
        
        self.cond_emb_y = torch.nn.Sequential(
            torch.nn.Linear(cdim, edim),
            torch.nn.LayerNorm(edim),
            torch.nn.GELU(),
        )

        self.cond_emb_t = torch.nn.Sequential(
            torch.nn.Linear(1, edim),
            torch.nn.LayerNorm(edim),
            torch.nn.GELU(),
        )

        self.attn = torch.nn.MultiheadAttention(embed_dim=edim, num_heads=num_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(edim)
          
        self.blocks = torch.nn.ModuleList()       
        for i in range(layers):
            self.blocks.append(ResidualBlock(w + edim, w))

        self.last_layer = torch.nn.Linear(w + edim, dim)

    def forward(self, x, y, t):
        y = self.cond_emb_y(y)
        t = self.cond_emb_t(t)
        y_residual = y
        t_residual = t

        y = self.ln(y)
        y, _ = self.attn(y, y, y)      # self-attention conditioning
        y += y_residual
        y_residual = y

        y = self.ln(y)
        t = self.ln(t)
        yt, _ = self.attn(y, t, t)      # cross-attention conditoning / time
        yt += y_residual + t_residual

        in_x = torch.cat([x, yt], -1)
        out = self.first_layer(in_x)

        for i in range(len(self.blocks)):
            out = self.blocks[i](out, yt)

        in_x = torch.cat([out, yt], -1)
        out = self.last_layer(in_x)

        return out   
        
class CombinedModel(pl.LightningModule):  
    def __init__(self, drift_model, score_model, hparams):
        super(CombinedModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.drift_model = drift_model
        self.score_model = score_model
    
    def forward(self, x, y, t):
        drift = self.drift_model(x, y, t)
        score = self.score_model(x, y, t)
        return drift, score
        
    def training_step(self, train_batch, batch_idx):
        tx, ty, uf, xs, ys = train_batch

        t0 = tx
        t1 = ty
        t = t0 + torch.rand(t0.shape[0], 1).to(self.device) * (t1 - t0)
        ts = (t - t0) / (t1 - t0)
        
        x0 = xs[:,self.hparams.cdim:]
        x1 = ys[:,self.hparams.cdim:]
        y0 = xs[:,:self.hparams.cdim]
        y1 = ys[:,:self.hparams.cdim]

        # Resample (x0, y0), (x1, y1) according to transport matrix
        M = torch.cdist(x0, x1) ** 2 + self.hparams.oty*torch.cdist(y0, y1) ** 2
        M = M / M.max()
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        pi = pot.emd(a, b, M.detach().cpu().numpy())
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x0.shape[0])
        i, j = np.divmod(choices, pi.shape[1])
        x0 = x0[i]
        x1 = x1[j]
        y0 = y0[i]
        y1 = y1[j]        

        # Construct estimate for velocity with conditional OT
        sigma_t = self.hparams.sigma * torch.sqrt(ts * (1 - ts))
        sigmay_t = self.hparams.sigmay * torch.sqrt(ts * (1 - ts))
        sigma_t_prime_over_sigma_t = (1 - 2 * ts) / (2 * ts * (1 - ts) + 1e-8)

        eps = torch.randn_like(x0).to(self.device)
        eps[:,0] = eps[:,0]*self.hparams.sigma_fac
        mu_t = x0 * (1 - ts) + x1 * ts
        xt = mu_t + sigma_t * eps

        epsy = torch.randn_like(y0).to(self.device)
        muy_t = y0 * (1 - ts) + y1 * ts
        yt = muy_t + sigmay_t * epsy
        
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + (x1 - x0)
        lambda_t = 2 * sigma_t / (self.hparams.sigma**2 + 1e-8)  

        vt, st = self(xt, yt, t)

        # compute losses with stationary kernel weighting
        err = torch.mean(torch.exp(-0.5*((y1 - y0) / self.hparams.sigmay)**2),dim=-1).unsqueeze(-1)
        drift_loss = torch.mean(err*(vt - ut)**2) 
        score_loss = torch.mean(err*(lambda_t[:, None] * st + eps) ** 2)
        
        loss = torch.sum(torch.stack((score_loss,drift_loss)))
    
        self.log("train_loss_drift", drift_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_loss_score", score_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=True)
        
        return loss 
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.drift_model.parameters()) + list(self.score_model.parameters()),
                                      lr=self.hparams.lr_init, weight_decay=self.hparams.wdecay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.n_epoch, self.hparams.lr_end)
                          
        return [optimizer], [scheduler] 
    

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model, cond, sigma=1.0, sigma_fac=1.0):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.sigma_fac = sigma_fac
        self.cond = cond

    # Drift
    def f(self, t, y):
        t = t.repeat(y.shape[0])[:, None]
        vt, st = self.model(y,self.cond,t)
        return vt + st

    # Diffusion
    def g(self, t, y):
        out = torch.ones_like(y) * self.sigma 
        out[:,0] = out[:,0]*self.sigma_fac
        return out

def ema_avg(averaged_model_parameter, model_parameter, weight_decay=0.9999): return \
    (1 - weight_decay) * averaged_model_parameter + weight_decay * model_parameter

def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ

def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ
