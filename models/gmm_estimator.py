import torch

def sqrt_SPD(cov):
    D, V = torch.linalg.eigh(cov)
    return V@torch.diag_embed(torch.sqrt(D + 1.0e-4))@V.permute(0, 2, 1)
def square_SPD(cov):
    D, V = torch.linalg.eigh(cov)
    return V@torch.diag_embed(D**2)@V.permute(0, 2, 1)
def distance_SPD(mu1, cov1, mu2, cov2):
    dist_mu = ((mu1-mu2)**2).sum(dim=1)
    sqrt_cov1 = sqrt_SPD(cov1)
    dist_cov = torch.einsum('nii -> n', sqrt_SPD(sqrt_cov1@cov2@sqrt_cov1))
    return dist_mu + dist_cov

class GMM_estimator(torch.nn.Module):
    def __init__(self, network, num_components=2, dim=2):
        super(GMM_estimator, self).__init__()
        self.network = network
        self.dim = dim
        self.num_components = num_components
        
    def forward(self, tau):
        outs = self.network(tau) # bs x -1
        # -> bs x num_components x dim
        # -> bs x num_components x dim x dim
        outs = outs.view(-1, self.num_components, self.dim + self.dim**2)
        mu = outs[:, :, :self.dim]
        cov = outs[:, :, self.dim:].view(-1, self.num_components, self.dim, self.dim)
        cov = (cov + cov.permute(0, 1, 3, 2))/2
        cov = square_SPD(cov.view(-1, self.dim, self.dim)).view(-1, self.num_components, self.dim, self.dim)
        return mu, cov
    
    def loss(self, pred_mu, pred_cov, label_mu, label_cov):
        bs, num_components, _ = pred_mu.size()
        assert num_components == 2
        assert label_mu.size(1) == 2
        
        dist1 = distance_SPD(pred_mu[:, 0], pred_cov[:, 0], label_mu[:, 0], label_cov[:, 0]) + distance_SPD(pred_mu[:, 1], pred_cov[:, 1], label_mu[:, 1], label_cov[:, 1])
        dist2 = distance_SPD(pred_mu[:, 0], pred_cov[:, 0], label_mu[:, 1], label_cov[:, 1]) + distance_SPD(pred_mu[:, 1], pred_cov[:, 1], label_mu[:, 0], label_cov[:, 0])
        idxs = dist1 < dist2
        loss = ((dist1[idxs]).sum() + (dist2[~idxs]).sum())/bs
        return loss
    
    def train_step(self, tau, mu, cov, optimizer, **kwargs):
        optimizer.zero_grad()
        mu_p, cov_p = self(tau)
        loss = self.loss(mu_p, cov_p, mu, cov)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, tau, mu, cov):
        mu_p, cov_p = self(tau)
        loss = self.loss(mu_p, cov_p, mu, cov)
        loss.backward()
        return {"loss": loss.item()}
    
    