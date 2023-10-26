import torch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import math

class Bezier:
    def __init__(self, dof, T, num_control_points, num_models=1, device='cpu'):
        self.dof = dof
        self.T = T
        self.num_control_points = num_control_points
        self.num_models = num_models
        self.device = device
        
        self.theta = torch.randn(num_models, num_control_points, dof).to(device) # (bs, n, dof)
        nfac = math.factorial(num_control_points-1)
        nCi = []
        for i in range(num_control_points):
            nCi.append(
                torch.tensor(
                    [nfac/(math.factorial(i)*math.factorial(num_control_points-i-1))]))
        self.nCi = torch.cat(nCi).to(device) # (n, )
        self.set_Riemannian_metrics()
        
    def basis_vector(self, t):
        # t : (bs, num_times)
        bs = len(t)
        basis_vec = []
        for i in range(self.num_control_points):
            basis = self.nCi[i]*(1-t/self.T)**(self.num_control_points-1-i) * (t/self.T)**(i)
            basis_vec.append(basis.view(bs, -1, 1))
        basis_vec = torch.cat(basis_vec, dim=-1)
        return basis_vec # (bs, num_times, n)
    
    def curve(self, t):
        B = self.basis_vector(t) # (bs, num_times, n)
        outs = B@self.theta # (bs, num_times, dof)
        return outs
    
    def LfD(self, q_demo_traj, T, out=False):
        # q_demo_traj : (bs, L, dof)
        bs, L, _ = q_demo_traj.size()
        assert bs == self.num_models
        t = torch.linspace(0, T, L).view(1, L).to(self.device)
        
        basis_vec = self.basis_vector(t) # (bs, L, n)
        ## LfD: q_demo_traj = basis_vec @ theta
        theta = torch.linalg.pinv(basis_vec)@q_demo_traj
        self.theta = theta
        if out:
            return theta
    
    def set_Riemannian_metrics(self, num=10000):
        # (bs, theta, theta) 
        t = torch.linspace(0, self.T, num).view(1, -1).to(self.device)
        b = self.basis_vector(t)
        self.H = (b.permute(0, 2, 1)@b)/num
        
def demo2bezier(q_traj, T, num_control_points):
    # q_traj : (bs, L, dof)
    bs, _, dof = q_traj.size()
    bezier = Bezier(dof, T, num_control_points, num_models=bs, device=q_traj.device)
    theta = bezier.LfD(q_traj, T, out=True)
    return theta.view(bs, -1)

def bezier2traj(theta, dof, T, traj_len=100):
    bs, Ndof = theta.size()
    num_control_points = int(Ndof/dof)
    theta = theta.view(bs, num_control_points, dof)
    bezier = Bezier(dof, T, num_control_points, num_models=bs, device=theta.device)
    bezier.theta = theta
    t = torch.linspace(0, T, traj_len).view(1, -1).to(theta)
    traj = bezier.curve(t)
    return traj
    
def bezier_Riemannian_metric(dof, T, num_control_points):
    model = Bezier(dof, T, num_control_points)
    model.set_Riemannian_metrics()
    return model.H