from .dataset import *
import copy
from mindspore.dataset import RandomSampler
import mindspore.dataset as ds


def build_dataloader(config):
    cfg = copy.deepcopy(config)
    dataset_cfg = cfg.pop('Dataset')
    dataname = dataset_cfg.pop('name')
    dataset = eval(dataname)(**dataset_cfg)

    dataloader_cfg = cfg.pop('DataLoader')
    sampler_cfg = dataloader_cfg.pop('sampler')
    sampler_name = sampler_cfg.pop('name')
    sampler = eval(sampler_name)(**sampler_cfg)

    dataloader = ds.GeneratorDataset(
        source=dataset, 
        column_names=['views_0', 'view_1', 'target', 'miss_m0', 'miss_m1']
    )
    dataloader = dataloader.batch(batch_size=dataloader_cfg['batch_size'])
    return dataloader