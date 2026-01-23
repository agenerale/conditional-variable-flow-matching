import torch
import pytorch_lightning as pl
import ot as pot
import numpy as np
import math

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale

class AdaLN(torch.nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = torch.nn.Linear(cond_dim, dim)
        self.shift = torch.nn.Linear(cond_dim, dim)
        
        # Initialize to identity transformation
        torch.nn.init.zeros_(self.scale.weight)
        torch.nn.init.ones_(self.scale.bias)
        torch.nn.init.zeros_(self.shift.weight)
        torch.nn.init.zeros_(self.shift.bias)
    
    def forward(self, x, cond):
        normalized = self.norm(x)
        scale = self.scale(cond)
        shift = self.shift(cond)
        
        while len(scale.shape) < len(x.shape):
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
            
        return normalized * (1 + scale) + shift

class FrequencyEmbedding(torch.nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, w=512, use_adaln=True):
        super(ResidualBlock, self).__init__()
        
        self.use_adaln = use_adaln
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, w),
            RMSNorm(w) if not use_adaln else torch.nn.Identity(),
            torch.nn.GELU(),
            torch.nn.Linear(w, out_dim),
        )
        
        if use_adaln:
            self.adaln1 = AdaLN(w, cond_dim)
            self.adaln2 = AdaLN(out_dim, cond_dim)
        else:
            self.ln = torch.nn.LayerNorm(out_dim)
            
        self.gelu = torch.nn.GELU()
        
    def forward(self, x, yt, cond=None):
        in_x = torch.cat([x, yt], -1)
        residual = x
        
        if self.use_adaln and cond is not None:
            hidden = self.net[0](in_x)
            hidden = self.adaln1(hidden, cond)
            hidden = self.net[2](hidden)
            out = self.net[3](hidden)
            
            out = out + residual
            out = self.adaln2(out, cond)
        else:
            out = self.net(in_x)
            out = out + residual
            out = self.ln(out)
            
        out = self.gelu(out)
        return out

class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, edim, layers=3, w=512, w_embed=32, num_heads=8, use_adaln=True):
        super(MLP, self).__init__()
        
        self.use_adaln = use_adaln
        
        self.first_layer = torch.nn.Sequential(
            torch.nn.Linear(edim + dim, w),
            RMSNorm(w) if not use_adaln else torch.nn.Identity(),
            torch.nn.GELU(),
        )
        
        if use_adaln:
            self.first_adaln = AdaLN(w, edim)
        
        self.cond_emb_y = torch.nn.Sequential(
            torch.nn.Linear(cdim, edim),
            torch.nn.LayerNorm(edim),
            torch.nn.GELU(),
        )

        self.freq_emb_t = FrequencyEmbedding(edim)
        self.cond_emb_t = torch.nn.Sequential(
            torch.nn.Linear(edim, edim),
            torch.nn.LayerNorm(edim),
            torch.nn.GELU(),
        )

        self.attn = torch.nn.MultiheadAttention(embed_dim=edim, num_heads=num_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(edim)
        
        self.cond_fusion = torch.nn.Sequential(
            torch.nn.Linear(edim * 2, edim),
            torch.nn.LayerNorm(edim),
            torch.nn.GELU(),
        )
          
        self.blocks = torch.nn.ModuleList()       
        for i in range(layers):
            self.blocks.append(ResidualBlock(w + edim, w, edim, use_adaln=use_adaln))

        self.last_layer = torch.nn.Linear(w + edim, dim)

    def forward(self, x, y, t):
        y = self.cond_emb_y(y)
        
        t_freq = self.freq_emb_t(t)
        t = self.cond_emb_t(t_freq)
        
        cond = self.cond_fusion(torch.cat([y, t], dim=-1))
        
        y = y.unsqueeze(1)
        t = t.unsqueeze(1)
        
        y_residual = y
        y = self.ln(y)
        y, _ = self.attn(y, y, y)
        y = y + y_residual
        
        y_residual = y
        y = self.ln(y)
        t = self.ln(t)
        yt, _ = self.attn(y, t, t)
        yt = yt + y_residual
        
        yt = yt.squeeze(1)
        
        in_x = torch.cat([x, yt], -1)
        
        if self.use_adaln:
            out = self.first_layer[0](in_x)
            out = self.first_adaln(out, cond)
            out = self.first_layer[2](out)
        else:
            out = self.first_layer(in_x)
        
        for i in range(len(self.blocks)):
            out = self.blocks[i](out, yt, cond if self.use_adaln else None)
        
        in_x = torch.cat([out, yt], -1)
        out = self.last_layer(in_x)
        
        return out

class CombinedModel(pl.LightningModule):  
    def __init__(self, hparams):
        super(CombinedModel, self).__init__()
        self.save_hyperparameters(hparams)
        
        self.drift_model = MLP(dim=hparams.ndim,
                              cdim=hparams.cdim,
                              edim=hparams.cond_emb,
                              layers=hparams.drift_layers,
                              w=hparams.drift_width,
                              num_heads=hparams.n_attn)
        self.score_model = MLP(dim=hparams.ndim,
                              cdim=hparams.cdim,
                              edim=hparams.cond_emb,
                              layers=hparams.score_layers,
                              w=hparams.score_width,
                              num_heads=hparams.n_attn)
        
    
    def forward(self, x, y, t):
        drift = self.drift_model(x, y, t)
        score = self.score_model(x, y, t)
        return drift, score
        
    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        B, T, F = x.shape
                
        t0 = torch.randint(0, T-1, (1,)).item()  
        t1 = t0 + 1
        
        x0 = x[:, t0, :]
        x1 = x[:, t1, :]

        y0 = x0[:,:self.hparams.cdim]
        y1 = x1[:,:self.hparams.cdim]                
        x0 = x0[:,self.hparams.cdim:]
        x1 = x1[:,self.hparams.cdim:]

        # Resample (x0, y0), (x1, y1) via transport matrix
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
        
        t = t0 + torch.rand(x0.shape[0], 1).to(self.device) * (t1 - t0)
        ts = (t - t0) / (t1 - t0)
        
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
        norm = torch.sum((y0 - y1) ** 2, dim=-1, keepdim=True)
        err = torch.exp(-0.5*(norm / (self.hparams.sigmay**2))).unsqueeze(-1)
        
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