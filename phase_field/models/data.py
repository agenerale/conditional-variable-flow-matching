import torch
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

            train_x, test_x, train_y, test_y, indx_train, indx_test = train_test_split(
                data[:,:-1,:], data[:,1:,:], np.arange(0,data.shape[0],1),
                test_size=0.2, random_state=random_seed)
            
            train_x = train_x[:,self.args.cutoff:,:(self.args.cdim+self.args.ndim)]
            train_y = train_y[:,self.args.cutoff:,:(self.args.cdim+self.args.ndim)]
            
            test_x = test_x[:,self.args.cutoff:,:(self.args.cdim+self.args.ndim)]
            test_y = test_y[:,self.args.cutoff:,:(self.args.cdim+self.args.ndim)]   

            num_traj = int(train_x.shape[0])
            train_xo = torch.from_numpy(train_x).float()
            train_yo = torch.from_numpy(train_y).float()
            test_xo = torch.from_numpy(test_x).float()
            test_yo = torch.from_numpy(test_y).float()
            
            self.train_xo = train_xo
            self.train_yo = train_yo
            self.test_xo = test_xo
            self.test_yo = test_yo

            # Train
            xs_train, ys_train, uf_train, tx_train, ty_train = self._split_data(
                train_xo, train_yo, num_traj,
                self.args.cutoff, self.args.ndim, self.args.cdim,
                )

            train_dataset = torch.utils.data.TensorDataset(tx_train, ty_train, uf_train, xs_train, ys_train)
            self.train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset, batch_size=self.args.batch_size, 
                            drop_last=False, shuffle=True)#, num_workers = self.args.num_workers)

            # Test
            xs_test, ys_test, uf_test, tx_test, ty_test = self._split_data(
                test_xo, test_yo, num_traj,
                self.args.cutoff, self.args.ndim, self.args.cdim
                )

            test_dataset = torch.utils.data.TensorDataset(tx_test, ty_test, uf_test, xs_test, ys_test)
            self.test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset, batch_size=self.args.batch_size,
                            drop_last=False, shuffle=True)#, num_workers = self.args.num_workers)

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

        uf = (train_yo[...,cdim:] - train_xo[...,cdim:])
            
        tx, ty = self._time_inc(train_xo, train_yo, cutoff)

        xs = train_xo.reshape(train_xo.shape[0]*train_xo.shape[1], ndim+cdim)
        ys = train_yo.reshape(train_yo.shape[0]*train_yo.shape[1], ndim+cdim)
        uf = uf.reshape(uf.shape[0]*uf.shape[1], ndim)
        
        return xs, ys, uf, tx, ty  

    def _time_inc(self,x,y,cutoff):        
                  
        tx = torch.linspace(0,100-cutoff-1,100-cutoff)[...,None] 
        ty = torch.linspace(1,100-cutoff,100-cutoff)[...,None]   
            
        tx = tx.repeat(x.shape[0], 1).to(self.device)
        ty = ty.repeat(y.shape[0], 1).to(self.device)

        return tx, ty

def norm_scaling(x):
    x = torch.from_numpy(x).float()
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s

    return x, m, s

def unit_scaling(x):
    x = torch.from_numpy(x).float()
    x_min = x.min(0)[0]
    x_max = x.max(0)[0]
    x = 2 * (x - x_min)/(x_max - x_min) - 1

    return x, x_min, x_max