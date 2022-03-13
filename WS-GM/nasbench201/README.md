## Experiments on NAS-Bench-201

The scripts for running experiments can be found in the `exp_scripts/` directory.

### Dataset preparation
1. Download the [NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) and save it under `./data` folder.

2. Install NasBench201 via pip. (Note: We use the `[2020-02-25]` version of the NAS-Bench-201 API. If you have the newer version installed, you might add `hp="200"` to `api.query_by_arch()` in `nasbench201/train_search.py`)
```
pip install nas-bench-201
```


### Running GM-NAS on NAS-Bench-201

#### GM+DARTS
The ckpts and logs will be saved to `./experiments/nasbench201/search-{script_name}-{seed}/`. For example, the ckpt dir would be `./experiments/nasbench201/search-darts-201-1/` for the command below. Each experiments are repeated with seed 0-3.
```
bash ws-so-darts-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --group bench --fs_opt 1 --restart final
bash ws-so-darts-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --group bench --fs_opt 1 --restart final --dataset cifar100
bash ws-so-darts-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --group bench --fs_opt 1 --restart final --dataset imagenet16-120
```

#### GM+SNAS
```
bash snas-ws-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --fs_opt 1 --group SNAS_NONE_OURS --restart final
bash snas-ws-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --fs_opt 1 --group SNAS_NONE_OURS --restart final --dataset cifar100
bash snas-ws-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --fs_opt 1 --group SNAS_NONE_OURS --restart final --dataset imagenet16-120
```

#### GM+RSPS
```
bash rsws-ws-201.sh --seed 0 --split_ckpts 20,40 --projection_warmup_epoch 50 --edge_crit grad --group RSWS --restart final --fs_opt 1
bash rsws-ws-201.sh --seed 0 --split_ckpts 20,40 --projection_warmup_epoch 50 --edge_crit grad --group RSWS --restart final --fs_opt 1 --dataset cifar100
bash rsws-ws-201.sh --seed 0 --split_ckpts 20,40 --projection_warmup_epoch 50 --edge_crit grad --group RSWS --restart final --fs_opt 1 --dataset imagenet16-120
```