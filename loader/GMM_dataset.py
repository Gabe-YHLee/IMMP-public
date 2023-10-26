import torch
from sklearn.mixture import GaussianMixture

class GMM_dataset(torch.utils.data.Dataset):
    def __init__(self, traj_ds, ae, tau_list=None):
        super(GMM_dataset, self).__init__()
        if tau_list is None:
            tau_list = [
                [0.8, 0.8], [0.4, 0.8], [0.0, 0.8], [-0.4, 0.8], [-0.8, 0.8],
                [0.8, 0.4], [0.4, 0.4], [0.0, 0.4], [-0.4, 0.4], [0.8, 0.4],
                [0.8, 0.0], [0.4, 0.0], [-0.4, 0.0], [-0.8, 0.0], [0.8, -0.4],
                [0.4, -0.4], [0.0, -0.4], [-0.4, -0.4], [-0.8, -0.4], [0.8, -0.8],
                [0.4, -0.8], [0.0, -0.8], [-0.4, -0.8],
            ]
        mu_list = []
        cov_list = []
        for i, tau_ in enumerate(tau_list):
            tau_ = torch.tensor(tau_, dtype=torch.float32)
            data = traj_ds.data[torch.norm(traj_ds.targets - tau_, dim=1) < 0.001]
            Z = ae.encode(data).detach().cpu()
            gm = GaussianMixture(n_components=2, random_state=0).fit(Z)
            mu = torch.tensor(gm.means_, dtype=torch.float32).unsqueeze(0) 
            cov = torch.tensor(gm.covariances_, dtype=torch.float32).unsqueeze(0)
            mu_list.append(mu)
            cov_list.append(cov)
        self.tau = torch.tensor(tau_list, dtype=torch.float32)
        self.mu = torch.cat(mu_list, dim=0)
        self.cov = torch.cat(cov_list, dim=0)
    
    def __len__(self):
        return len(self.tau)

    def __getitem__(self, idx):
        tau = self.tau[idx]
        mu = self.mu[idx]
        cov = self.cov[idx]
        return tau, mu, cov