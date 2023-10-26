from torch.utils import data

from loader.Toy_dataset import Toy
from loader.Toy_dataset2 import Toy2
from loader.GMM_dataset import GMM_dataset

from models import load_pretrained

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader

def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == "Toy":
        dataset = Toy(**data_dict)
    elif name == "Toy2":
        dataset = Toy2(**data_dict)
    elif name == "GMM":
        traj_ds = Toy2(**data_dict['Toy2'])
        ae, _ = load_pretrained(**data_dict['pretrained'])
        dataset = GMM_dataset(traj_ds=traj_ds, ae=ae)
    return dataset