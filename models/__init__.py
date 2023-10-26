import os
from omegaconf import OmegaConf
import torch

from models.ae import (
    AE,
    IRAE,
    cAE,
    cIRAE
)

from models.modules import (
    FC_vec,
)

from models.gmm_estimator import GMM_estimator

def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    return net

def get_ae(model_cfg, **kwargs):
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    if arch == 'ae':
        init_final_projection = model_cfg.get('init_final_projection', False)
        init_projection = model_cfg.get('init_projection', False) 
        init = model_cfg.get('init', [0.8, 0.8])
        goal = model_cfg.get('goal', [-0.8, -0.8])
        dof = model_cfg.get('dof', 2)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = AE(
            encoder, 
            decoder, 
            init_final_projection=init_final_projection, 
            init_projection=init_projection,
            init=init,
            goal=goal,
            dof=dof)
    elif arch == 'cae':
        final_projection = model_cfg.get('final_projection', False)
        goal = model_cfg.get('goal', [-0.8, -0.8])
        dof = model_cfg.get('dof', 2)
        tau_dim = model_cfg.get('tau_dim', 2)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim+tau_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = cAE(
            encoder, 
            decoder, 
            final_projection=final_projection, 
            goal=goal,
            dof=dof)
    elif arch == 'irae':
        metric = model_cfg.get("metric", "identity")
        iso_reg = model_cfg.get("iso_reg", 1.0)
        dof = model_cfg.get('dof', 2)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        dim = model_cfg.get("dim", None)
        T = model_cfg.get("T", None)
        num_control_points = model_cfg.get("num_control_points", None)
        init_final_projection = model_cfg.get('init_final_projection', False)
        init_projection = model_cfg.get('init_projection', False) 
        init = model_cfg.get('init', [0.8, 0.8])
        goal = model_cfg.get('goal', [-0.8, -0.8])
        model = IRAE(
            encoder, 
            decoder, 
            init_final_projection=init_final_projection,
            init_projection=init_projection,
            iso_reg=iso_reg, 
            metric=metric,
            dim=dim,
            T=T,
            num_control_points=num_control_points, 
            init=init,
            goal=goal,
            dof=dof,
            **kwargs)
    elif arch == 'cirae':
        metric = model_cfg.get("metric", "identity")
        iso_reg = model_cfg.get("iso_reg", 1.0)
        dof = model_cfg.get('dof', 2)
        dim = model_cfg.get("dim", 2)
        T = model_cfg.get("T", 1)
        num_control_points = model_cfg.get("num_control_points", 10)
        final_projection = model_cfg.get('final_projection', False)
        goal = model_cfg.get('goal', [-0.8, -0.8])
        dof = model_cfg.get('dof', 2)
        tau_dim = model_cfg.get('tau_dim', 2)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim+tau_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = cIRAE(
            encoder, 
            decoder, 
            final_projection=final_projection,
            iso_reg=iso_reg, 
            metric=metric,
            dim=dim,
            T=T,
            num_control_points=num_control_points, 
            goal=goal,
            dof=dof,
            **kwargs)
    return model

def get_gmm_estimator(model_cfg, **kwargs):
    dim = model_cfg['dim']
    in_dim = model_cfg.get('in_dim', 2)
    num_components = model_cfg.get('num_components', 2)
    out_dim = num_components*(dim + dim**2)
    network = get_net(in_dim=in_dim, out_dim=out_dim, **model_cfg["network"])
    model = GMM_estimator(
        network=network,
        dim=dim
    )
    return model

def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(model_dict, **kwargs)
    return model

def _get_model_instance(name):
    try:
        return {
            "vae": get_ae,
            "irvae": get_ae,
            "ae": get_ae,
            "irae": get_ae,
            "cae": get_ae,
            "cirae": get_ae,
            "gmm_estimator": get_gmm_estimator
        }[name]
    except:
        raise ("Model {} not available".format(name))

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
   
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    
    model = get_model(cfg, **kwargs)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    if kwargs.get('eval', False):
        pretrained_dict = ckpt
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)
    else:   
        model.load_state_dict(ckpt)
    
    return model, cfg