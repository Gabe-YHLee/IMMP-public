import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.utils import label_to_color, figure_to_array, PD_metric_to_ellipse

from geometry import (
    relaxed_distortion_measure,
)

from bezier import bezier_Riemannian_metric
from scipy.optimize import linear_sum_assignment

import umap
import copy

from matplotlib.patches import Circle

from bezier import demo2bezier, bezier2traj
from sklearn.mixture import GaussianMixture

class AE(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder, 
        init_final_projection=False, 
        init_projection=False,
        dof=None,
        init=[0.8, 0.8],
        goal=[-0.8, -0.8]
        ):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_final_projection = init_final_projection
        self.init_projection = init_projection
        
        if dof is None:
            self.dof = 2
        else:
            self.dof =dof 
        
        if self.init_final_projection:
            self.init = torch.tensor(init, dtype=torch.float32)
            self.goal = torch.tensor(goal, dtype=torch.float32)
        elif self.init_projection:
            self.init = torch.tensor(init, dtype=torch.float32)
            
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        if self.init_final_projection:
            bs = len(z)
            theta = self.decoder(z)
            theta = theta.view(bs, -1, self.dof)
            theta[:, 0, :] = self.init.to(z).repeat(bs, 1)
            theta[:, -1, :] = self.goal.to(z).repeat(bs, 1)
            return theta.view(bs, -1)
        elif self.init_projection:
            bs = len(z)
            theta = self.decoder(z)
            theta = theta.view(bs, -1, self.dof)
            theta[:, 0, :] = self.init.to(z).repeat(bs, 1)
            return theta.view(bs, -1)
        else:
            return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, **kwargs):
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}

    def eval_step(self, dl, **kwargs):
        device = kwargs["device"]

        ## ENV PARAMS
        env = dl.dataset.env
        bezier_params = dl.dataset.bezier_params
        dof = bezier_params['dof']
        T = bezier_params['T']
        xlim = env['xlim']
        ylim = env['ylim']
        init = env['start_point_pos']
        goal = env['final_point_pos']
        obstacles = env['obstacles']
        
        x = dl.dataset.data
        Z = self.encode(x.to(device)).detach().cpu()
        label_unique = torch.unique(dl.dataset.targets)
        num_calsses = len(label_unique)
        gm = GaussianMixture(n_components=num_calsses, random_state=0).fit(Z)
        sample_z, sample_y = gm.sample(100)
        
        sample_x = self.decode(torch.tensor(sample_z, dtype=torch.float32).to(device))
        sampled_trajs = bezier2traj(
            sample_x, dof, T, traj_len=500).detach().cpu()
        collision_results = dl.dataset.check_coliision(sampled_trajs)
        collision_ratio = sum(collision_results)/len(collision_results)
        
        return {'collision_ratio_': collision_ratio}
    
    def visualization_step(self, dl, **kwargs):
        device = kwargs["device"]

        ## ENV PARAMS
        env = dl.dataset.env
        bezier_params = dl.dataset.bezier_params
        dof = bezier_params['dof']
        T = bezier_params['T']
        xlim = env['xlim']
        ylim = env['ylim']
        init = env['start_point_pos']
        goal = env['final_point_pos']
        obstacles = env['obstacles']
        
        ## RECONSTRUCTION FIGURE
        x = dl.dataset.data
        z = self.encode(x.to(device))
        recon = self.decode(z).detach().cpu()
        
        x_traj = bezier2traj(x, dof, T, traj_len=500)
        recon_traj = bezier2traj(recon, dof, T, traj_len=500)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        Z = self.encode(x.to(device)).detach().cpu()

        for i in [0, 2]:
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].set_aspect('equal')
            axs[i].axis('off')
            axs[i].scatter(init[0], init[1], marker='s', s=150, c='k')
            axs[i].scatter(goal[0], goal[1], marker='*', s=300, c='k')
            
            for pos, rad in zip(obstacles['pos'], obstacles['rad']):
                theta = np.linspace(0, 2*np.pi, 100)
                xx = np.array(pos[0]) + np.cos(theta) * rad
                yy = np.array(pos[1]) + np.sin(theta) * rad
                # Obs = Circle(xy=pos, radius=rad, color='tab:orange')
                # axs[i].add_patch(Obs)
                axs[i].plot(xx, yy, color='tab:orange')

        y = dl.dataset.targets.numpy()
        for traj, label in zip(x_traj, y):
            color = label_to_color(np.array(label).reshape(1))
            axs[0].plot(traj[:, 0], traj[:, 1], c=color/255)

        axs[1].scatter(Z[:, 0], Z[:, 1], marker='x', s=100, c=label_to_color(y)/255)
        axs[1].set_aspect('equal')
        axs[1].axis('off')
       
        for traj, label in zip(recon_traj, y):
            color = label_to_color(np.array(label).reshape(1))
            axs[2].plot(traj[:, 0], traj[:, 1], c=color/255)

        ## SAMPLING
        fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5))
        label_unique = torch.unique(dl.dataset.targets)
        
        axs2[0].scatter(Z[:, 0], Z[:, 1], marker='x', s=100, c=label_to_color(y)/255)
        axs2[0].set_aspect('equal')
        axs2[0].axis('off')
        
        num_calsses = len(label_unique)
        gm = GaussianMixture(n_components=num_calsses, random_state=0).fit(Z)
        sample_z, sample_y = gm.sample(100)
        axs2[0].scatter(sample_z[:, 0], sample_z[:, 1], c=label_to_color(10-sample_y)/255, marker='*')
        
        # mu = gm.means_
        # cov = gm.covariances_
        
        # for i, (mu, cov) in enumerate(zip(gm.means_, gm.covariances_)):
        #     tempZ = sample_z[sample_y == i]
        #     x = np.linspace(tempZ.min(axis=0)[0], tempZ.max(axis=0)[0], 100)
        #     y = np.linspace(tempZ.min(axis=0)[1], tempZ.max(axis=0)[1], 100)
        #     xx, yy = np.meshgrid(x, y) 
        #     delta = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1) - mu.reshape(1, -1)

        #     zz = np.exp(np.diagonal(
        #         -(delta@np.linalg.inv(cov)@delta.transpose())
        #     )/2)

        #     contour = axs2[0].contour(
        #         xx.reshape(100, 100), 
        #         yy.reshape(100, 100), 
        #         zz.reshape(100, 100),
        #         levels=[0.85, 0.95], #0.75, 0.35, 0.55, 
        #         colors=label_to_color(10-np.array([label]))/255,
        #         linewidths=2,
        #         # linestyles='--',
        #         )
            
        for i in [1]:
            axs2[i].set_xlim(xlim)
            axs2[i].set_ylim(ylim)
            axs2[i].set_aspect('equal')
            axs2[i].axis('off')
            axs2[i].scatter(init[0], init[1], marker='s', s=150, c='k')
            axs2[i].scatter(goal[0], goal[1], marker='*', s=300, c='k')
            
            for pos, rad in zip(obstacles['pos'], obstacles['rad']):
                theta = np.linspace(0, 2*np.pi, 100)
                xx = np.array(pos[0]) + np.cos(theta) * rad
                yy = np.array(pos[1]) + np.sin(theta) * rad
                # Obs = Circle(xy=pos, radius=rad, color='tab:orange')
                # axs[i].add_patch(Obs)
                axs2[i].plot(xx, yy, color='tab:orange')
                
        sample_x = self.decode(torch.tensor(sample_z, dtype=torch.float32).to(device))
        sampled_trajs = bezier2traj(
            sample_x, dof, T, traj_len=500).detach().cpu()
        for traj, label in zip(sampled_trajs, sample_y):
            color = label_to_color(10-np.array(label).reshape(1))
            axs2[1].plot(traj[:, 0], traj[:, 1], c=color/255)
            
        result_dict = {
            'recon#': fig,
            'sample#': fig2
        }
        return result_dict 

class cAE(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder, 
        final_projection=False, 
        dof=2,
        goal=[-0.8, -0.8]
        ):
        super(cAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.final_projection = final_projection

        self.dof = dof 
        if self.final_projection:
            self.goal = torch.tensor(goal, dtype=torch.float32)
            
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, tau):
        z_cat = torch.cat([z, tau], dim=1)
        if self.final_projection:
            bs = len(z_cat)
            theta = self.decoder(z_cat)
            theta = theta.view(bs, -1, self.dof)
            theta[:, -1, :] = self.goal.to(z).repeat(bs, 1)
            theta[:, 0, :] = tau
            return theta.view(bs, -1)
        else:
            return self.decoder(z_cat)

    def train_step(self, x, tau, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encode(x)
        recon = self.decode(z, tau)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, tau, **kwargs):
        z = self.encode(x)
        recon = self.decode(z, tau)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}

    def visualization_step(self, dl, **kwargs):
        device = kwargs["device"]

        ## ENV PARAMS
        env = dl.dataset.env
        bezier_params = dl.dataset.bezier_params
        dof = bezier_params['dof']
        T = bezier_params['T']
        xlim = env['xlim']
        ylim = env['ylim']
        goal = env['final_point_pos']
        obstacles = env['obstacles']
        
        ## RECONSTRUCTION FIGURE
        x = dl.dataset.data
        tau = dl.dataset.targets
        z = self.encode(x.to(device))
        recon = self.decode(z, tau.to(device)).detach().cpu()
        
        x_traj = bezier2traj(x, dof, T, traj_len=500)
        recon_traj = bezier2traj(recon, dof, T, traj_len=500)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        Z = self.encode(x.to(device)).detach().cpu()
        
        for i in [0, 2]:
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].set_aspect('equal')
            axs[i].axis('off')
            axs[i].scatter(goal[0], goal[1], marker='*', s=300, c='k')
            
            for pos, rad in zip(obstacles['pos'], obstacles['rad']):
                theta = np.linspace(0, 2*np.pi, 100)
                xx = np.array(pos[0]) + np.cos(theta) * rad
                yy = np.array(pos[1]) + np.sin(theta) * rad
                # Obs = Circle(xy=pos, radius=rad, color='tab:orange')
                # axs[i].add_patch(Obs)
                axs[i].plot(xx, yy, color='tab:orange')

        # y = dl.dataset.targets.numpy()
        for traj in x_traj:
            axs[0].plot(traj[:, 0], traj[:, 1], c='tab:blue')

        axs[1].scatter(Z[:, 0], Z[:, 1], marker='x', s=100, c='tab:blue')
        axs[1].set_aspect('equal')
        axs[1].axis('off')
       
        for traj in recon_traj:
            axs[2].plot(traj[:, 0], traj[:, 1], c='tab:blue')

        # ## SAMPLING
        # fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5))
        # label_unique = torch.unique(dl.dataset.targets)
        
        # axs2[0].scatter(Z[:, 0], Z[:, 1], marker='x', s=100, c=label_to_color(y)/255)
        # axs2[0].set_aspect('equal')
        # axs2[0].axis('off')
        
        # num_calsses = len(label_unique)
        # gm = GaussianMixture(n_components=num_calsses, random_state=0).fit(Z)
        # sample_z, sample_y = gm.sample(100)
        # axs2[0].scatter(sample_z[:, 0], sample_z[:, 1], c=label_to_color(10-sample_y)/255, marker='*')
            
        # for i in [1]:
        #     axs2[i].set_xlim(xlim)
        #     axs2[i].set_ylim(ylim)
        #     axs2[i].set_aspect('equal')
        #     axs2[i].axis('off')
        #     axs2[i].scatter(goal[0], goal[1], marker='*', s=300, c='k')
            
        #     for pos, rad in zip(obstacles['pos'], obstacles['rad']):
        #         theta = np.linspace(0, 2*np.pi, 100)
        #         xx = np.array(pos[0]) + np.cos(theta) * rad
        #         yy = np.array(pos[1]) + np.sin(theta) * rad
        #         # Obs = Circle(xy=pos, radius=rad, color='tab:orange')
        #         # axs[i].add_patch(Obs)
        #         axs2[i].plot(xx, yy, color='tab:orange')
                
        # sample_x = self.decode(torch.tensor(sample_z, dtype=torch.float32).to(device))
        # sampled_trajs = bezier2traj(
        #     sample_x, dof, T, traj_len=500).detach().cpu()
        # for traj, label in zip(sampled_trajs, sample_y):
        #     color = label_to_color(10-np.array(label).reshape(1))
        #     axs2[1].plot(traj[:, 0], traj[:, 1], c=color/255)
            
        result_dict = {
            'recon#': fig,
            # 'sample#': fig2
        }
        return result_dict 
 
class IRAE(AE):
    def __init__(
        self, 
        encoder, 
        decoder, 
        iso_reg=1.0, 
        metric='identity', 
        dim=None,
        gamma=1e-6,
        num_points=None,
        num_control_points=None,
        T=None,
        dt=None,
        dl=None, 
        init_final_projection=False, 
        init_projection=False,
        dof=None,
        init=[0.8, 0.8],
        goal=[-0.8, -0.8]
    ):
        super(IRAE, self).__init__(
            encoder, 
            decoder, 
            init_final_projection=init_final_projection, 
            dof=dof,
            init=init,
            goal=goal
            )
        self.iso_reg = iso_reg
        self.metric = metric
        self.dim = dim
        self.num_points = num_points
        self.num_control_points = num_control_points
        self.T = T
        self.dt =dt
        self.gamma = gamma
        
        if metric == 'bezier':
            assert dim is not None
            assert T is not None
            assert num_control_points is not None
            self.H = bezier_Riemannian_metric(dim, T, num_control_points) # (1, -1, -1)
        else:
            self.H = None
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encode(x)
        recon = self.decode(z)
        recon_loss = ((x - recon)**2).view(len(x), -1).mean(dim=1)
        
        dict_metric = {}
        iso_loss = relaxed_distortion_measure(
            self.decode, 
            z, 
            eta=0.2, 
            metric=self.metric,
            dim=self.dim,
            gamma=self.gamma,
            num_points=self.num_points,
            T = self.T,
            dt = self.dt,
            H = self.H,
            **dict_metric
            )
          
        loss = recon_loss.mean() + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item()}

class cIRAE(cAE):
    def __init__(
        self, 
        encoder, 
        decoder, 
        iso_reg=1.0, 
        metric='identity', 
        dim=None,
        gamma=1e-6,
        num_points=None,
        num_control_points=None,
        T=None,
        dt=None,
        dl=None, 
        final_projection=False, 
        dof=None,
        goal=[-0.8, -0.8]
    ):
        super(cIRAE, self).__init__(
            encoder, 
            decoder, 
            final_projection=final_projection, 
            dof=dof,
            goal=goal
            )
        self.iso_reg = iso_reg
        self.metric = metric
        self.dim = dim
        self.num_points = num_points
        self.num_control_points = num_control_points
        self.T = T
        self.dt =dt
        self.gamma = gamma
        
        if metric == 'bezier':
            assert dim is not None
            assert T is not None
            assert num_control_points is not None
            self.H = bezier_Riemannian_metric(dim, T, num_control_points) # (1, -1, -1)
        else:
            self.H = None
            
    def train_step(self, x, tau, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encode(x)
        recon = self.decode(z, tau)
        recon_loss = ((x - recon)**2).view(len(x), -1).mean(dim=1)
        
        dict_metric = {}
        def wrapper(z):
            return self.decode(z, tau)
            
        iso_loss = relaxed_distortion_measure(
            wrapper, 
            z, 
            eta=0.2, 
            metric=self.metric,
            dim=self.dim,
            gamma=self.gamma,
            num_points=self.num_points,
            T = self.T,
            dt = self.dt,
            H = self.H,
            **dict_metric
            )
          
        loss = recon_loss.mean() + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item()}
    