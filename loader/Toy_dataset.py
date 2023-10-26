import torch
import numpy as np
import os, sys
from omegaconf import OmegaConf
from bezier import Bezier, bezier2traj, demo2bezier 

class Toy(torch.utils.data.Dataset):
    def __init__(self,
        root,
        num_control_points=10,
        T=1,
        split='training',
        **kwargs):

        super(Toy, self).__init__()

        data = []
        targets = []
        for file_ in os.listdir(root):
            if file_.endswith('.npy'):
                traj_data = np.load(os.path.join(root, file_))
                data.append(torch.tensor(traj_data[0], dtype=torch.float32).unsqueeze(0))
                targets.append(torch.tensor(int(file_[5]), dtype=torch.long).view(-1))
            elif file_.endswith('yaml'):
                cfg = OmegaConf.load(os.path.join(root, file_))
        data = torch.cat(data, dim=0)
        targets = torch.cat(targets, dim=0)
        
        data = demo2bezier(data, T=T, num_control_points=num_control_points)        
        
        data.size(), targets.size()

        xlim = cfg['xlim']
        ylim = cfg['ylim']
        start_point_pos = cfg['start_point_pos']
        final_point_pos = cfg['final_point_pos']
        obstacles = cfg['obstacles']

        env = {
            'xlim': xlim,
            'ylim': ylim,
            'start_point_pos': start_point_pos,
            'final_point_pos': final_point_pos,
            'obstacles': obstacles
        }
        
        bezier_params = {
            'T': T,
            'num_control_points': num_control_points,
            'dof': 2          
        }
        
        self.env = env
        self.bezier_params = bezier_params
        
        self.data = data
        self.targets = targets

        print(f"Toy split {split} | {self.data.size()}")
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y
    
    def check_coliision(self, traj):
        pos = self.env['obstacles']['pos']
        rad =  self.env['obstacles']['rad']
        no = len(rad)
        pos = torch.tensor(pos, dtype=torch.float32).view(no, 2).to(traj) # (no, 2)
        rad = torch.tensor(rad, dtype=torch.float32).view(no, 1).to(traj) # (no, )
        bs = len(traj)
        
        # traj : (bs, L, dof)
        dist2center = ((pos.view(1, 1, no, 2) - traj.view(bs, -1, 1, 2))**2).sum(dim=-1) # (bs, L, no)
        temp = (dist2center - rad.view(1, 1, -1)**2).view(bs, -1)
        outputs = torch.min(temp, dim=1).values # (bs, no)
        return outputs < 0
        
        
        