from torch.utils.data import DataLoader

import data.transform as alt_trans
import torchvision.transforms as tvs_trans
import distributed as alt_dist
from data.dataset import SingleLabelDataset
from data.sampler import DistributedSampler
from filereader import DirectReader
from data.autoaugment import ImageNetPolicy


def get_train_loader(conf, root_dir, autoaug=0):
    if autoaug:
        print("data auto")
        transform = tvs_trans.Compose([
            tvs_trans.RandomResizedCrop(size=224),
            tvs_trans.RandomHorizontalFlip(),
            ImageNetPolicy(),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = tvs_trans.Compose([
            tvs_trans.RandomResizedCrop(size=224),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            tvs_trans.ToTensor(),
            alt_trans.FancyPCA(),
            tvs_trans.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    dataset = SingleLabelDataset(
        imglist=root_dir+'/ILSVRC/Data/meta/train.txt',
        root=root_dir+'/ILSVRC/Data/CLS-LOC/train',

        transform=transform,
        skip_broken=False,
        reader=DirectReader(),
    )


    sampler = DistributedSampler(dataset, shuffle=True)
    world_size = alt_dist.get_world_size()
    print("conf.batch_size // world_size", conf.batch_size)
    print("conf.batch_size // world_size", world_size)
    print("conf.batch_size // world_size", conf.batch_size // world_size)
    loader = DataLoader(
        dataset,
        batch_size=conf.batch_size // world_size,
        num_workers=conf.num_worker,
        sampler=sampler,
        pin_memory=False,
        drop_last=False,
    )
    return loader


def get_valid_loader(conf, root_dir):
    transform = tvs_trans.Compose([
        tvs_trans.Resize(256),
        tvs_trans.CenterCrop(224),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    dataset = SingleLabelDataset(
        imglist=root_dir+'/ILSVRC/Data/meta/val.txt',
        root=root_dir+'/ILSVRC/Data/CLS-LOC/val',

        transform=transform,
        skip_broken=False,
        reader=DirectReader(),
    )


    sampler = DistributedSampler(dataset, shuffle=False, psudo_index=-1)
    world_size = alt_dist.get_world_size()
    loader = DataLoader(
        dataset,
        batch_size=conf.batch_size // world_size,
        num_workers=conf.num_worker,
        sampler=sampler,
        pin_memory=False,
        drop_last=False,
    )
    return loader
