import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os

class SpinodalDataModule(pl.LightningDataModule):
    ''' a lightning dataloader '''

    def __init__(
            self, 
            filename: str=None, 
            args = None, 
            device =None,
        ):
        super().__init__()
        self.filename = filename
        self.args = args
        self.device = device
        self.train_loader = None
        self.test_loader = None      
        self.data_loaded = False

    def prepare_data(self):

        random_seed = self._set_seed()

        if not self.data_loaded:

            # load data
            with h5py.File(self.filename, "r") as f:
                print("Keys: %s" % f.keys())
                pcs = f['scores'][()]
                parameters = f['parameters'][()]
            f.close()

            parameters[:,:2] = np.log(parameters[:,:2]) # return to [-1,1] range
            params, params_min, params_max = unit_scaling(parameters)

            filename_scale = './data/scaling_params.h5'

            if self.args.scaling:
                print('Standardize gradient')
                pcs, pcs_m, pcs_s = norm_scaling(pcs)
                pcs = pcs.detach().cpu().numpy()
                pcs_m = pcs_m.detach().cpu().numpy()
                pcs_s = pcs_s.detach().cpu().numpy()
                if not os.path.isfile(filename_scale):
                    with h5py.File(filename_scale, 'w') as f:
                        f.create_dataset('params_min', data=params_min)
                        f.create_dataset('params_max', data=params_max)
                        f.create_dataset('pcs_m', data=pcs_m)
                        f.create_dataset('pcs_s', data=pcs_s)
                    f.close()
            else:
                print('Raw gradient')
                if not os.path.isfile(filename_scale):
                    with h5py.File(filename_scale, 'w') as f:
                        f.create_dataset('params_min', data=params_min)
                        f.create_dataset('params_max', data=params_max)
                    f.close()

            params = np.tile(params[:,None,:],(1,pcs.shape[1],1))
            data = np.concatenate((params,pcs),axis=-1)
            
            train_x, test_x = train_test_split(
                data,
                test_size=0.2, random_state=random_seed)
            
            self.train_x = torch.from_numpy(train_x[:,self.args.cutoff:,:(self.args.cdim+self.args.ndim)])
            self.test_x = torch.from_numpy(test_x[:,self.args.cutoff:,:(self.args.cdim+self.args.ndim)])
            
            # shuffle across time
            samples, times, features = train_x.shape
            train_x_shuffled = train_x.copy()
            for t in range(times):
                perm = np.random.permutation(samples)
                train_x_shuffled[:, t, :] = train_x[perm, t, :]
            
            train_dataset = torch.utils.data.TensorDataset(self.train_x)
            test_dataset = torch.utils.data.TensorDataset(self.test_x)
            
            self.train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset, batch_size=self.args.batch_size, 
                            drop_last=False, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset, batch_size=self.args.batch_size,
                            drop_last=False, shuffle=True)

            self.data_loaded = True

            print('Data Module Loaded')

    def train_dataloader(self):
        return self.train_loader
    
    #def val_dataloader(self):
    #    return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def _set_seed(seed: int=2024):
        # standardizing randomness
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed
    
    def _split_data(self, train_xo, train_yo, num_traj, cutoff, ndim, cdim):

        tx, ty = self._time_inc(train_xo, train_yo, cutoff)

        xs = train_xo.reshape(train_xo.shape[0]*train_xo.shape[1], ndim+cdim)
        ys = train_yo.reshape(train_yo.shape[0]*train_yo.shape[1], ndim+cdim)
        
        index_x = np.random.choice(np.arange(xs.shape[0]), size=xs.shape[0])
        index_y = np.random.choice(np.arange(ys.shape[0]), size=ys.shape[0])
        
        tx = tx[index_x]
        xs = xs[index_x]
        ty = ty[index_y]
        ys = ys[index_y]
        
        return xs, ys, tx, ty  

    def _time_inc(self,x,y,cutoff):        
                  
        tx = torch.linspace(0,100-cutoff-1,100-cutoff)[...,None] 
        ty = torch.linspace(1,100-cutoff,100-cutoff)[...,None]   
            
        tx = tx.repeat(x.shape[0], 1).to(self.device)
        ty = ty.repeat(y.shape[0], 1).to(self.device)

        return tx, ty
    
class RandomTimeFrameDataset(Dataset):
    def __init__(self, train_x):
        """
        Args:
            train_x (np.ndarray or torch.Tensor): Array of shape [samples, time frames, features].
        """
    
        if isinstance(train_x, np.ndarray):
            train_x = torch.from_numpy(train_x)
            
        self.train_x = train_x
        self.samples, self.times, self.features = train_x.shape
        
    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        t = np.random.randint(0, self.times - 1)
        x0 = self.train_x[index, t, :]
        x1 = self.train_x[index, t+1, :]
        t0 = torch.tensor(t)
        t1 = t0 + 1
        return x0, x1, t0, t1

def norm_scaling(x):
    
    # flatten = False
    # if len(x.shape) > 2:
    #     b, t, f = x.shape
    #     x = x.reshape(b * t, f)
    #     flatten = True
    
    x = torch.from_numpy(x).float()
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    
    # if flatten: x = x.reshape(b, t, f)

    return x, m, s

def unit_scaling(x):   
    x = torch.from_numpy(x).float()
    x_min = x.min(0)[0]
    x_max = x.max(0)[0]
    x = 2 * (x - x_min)/(x_max - x_min) - 1

    return x, x_min, x_max