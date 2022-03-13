## Requirements
* python packages
    * Python >= 3.7
    * PyTorch >= 1.5
    * tensorboard == 2.0.1
    * gpustat

* Some codes are borrowed from **DARTS-PT** ([https://github.com/ruocwang/darts-pt]) 





## NASBench-201 space


### Dataset preparation
1. Download the [NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) and save it under `./data` folder.

2. Install NasBench201 via pip. (Note: We use the `[2020-02-25]` version of the NAS-Bench-201 API. If you have the newer version installed, you might add `hp="200"` to `api.query_by_arch()` in `nasbench201/train_search.py`)
```
pip install nas-bench-201
```

### Running GM-NAS on NASBench-201

GM + DARTS (2nd) :
```shell
cd exp_scripts
bash ws-so-darts-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --group bench --fs_opt 1 --restart final
bash ws-so-darts-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --group bench --fs_opt 1 --restart final --dataset cifar100
bash ws-so-darts-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --group bench --fs_opt 1 --restart final --dataset imagenet16-120
```

GM + SNAS :
```shell
cd exp_scripts
bash snas-ws-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --fs_opt 1 --group SNAS_NONE_OURS --restart final
bash snas-ws-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --fs_opt 1 --group SNAS_NONE_OURS --restart final --dataset cifar100
bash snas-ws-201.sh --seed 0 --split_ckpts 15,30 --projection_warmup_epoch 30 --edge_crit grad --fs_opt 1 --group SNAS_NONE_OURS --restart final --dataset imagenet16-120
```

GM + RSPS :
```shell
cd exp_scripts
bash rsws-ws-201.sh --seed 0 --split_ckpts 20,40 --projection_warmup_epoch 50 --edge_crit grad --group RSWS --restart final --fs_opt 1
bash rsws-ws-201.sh --seed 0 --split_ckpts 20,40 --projection_warmup_epoch 50 --edge_crit grad --group RSWS --restart final --fs_opt 1 --dataset cifar100
bash rsws-ws-201.sh --seed 0 --split_ckpts 20,40 --projection_warmup_epoch 50 --edge_crit grad --group RSWS --restart final --fs_opt 1 --dataset imagenet16-120
```




## DARTS space


### Search
Few-Shot + DARTS (1st) :
```shell
cd exp_scripts
bash darts-sota-fs-res.sh --split_num 7 --seed 0
```

GM + DARTS (1st) :
```shell
cd exp_scripts
bash darts-sota-ws-res.sh --edge_crit grad --seed 0
```

Few-Shot + DARTS (2nd) :
```shell
cd exp_scripts
bash darts-so-sota-fs-res.sh --split_num 7 --seed 0
```

GM + DARTS (2nd) :
```shell
cd exp_scripts
bash darts-so-sota-ws-res.sh --edge_crit grad --seed 0
```

Few-Shot + SNAS :
```shell
cd exp_scripts
bash snas-fs-darts-res.sh --split_num 7 --seed 0
```

GM + SNAS :
```shell
cd exp_scripts
bash snas-ws-darts-res.sh --edge_crit grad --seed 0
```
Note that in our experiments the seed is set to be 0,1,2,3


### Evaluation
```shell
cd exp_scripts
bash eval.sh --arch genotype
```
for example:
```shell
cd exp_scripts
bash eval.sh --arch search_darts_sota_ws_res_s5_2_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_5
bash eval.sh --arch search_darts_so_sota_ws_s5_1_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_4
bash eval.sh --arch search_snas_ws_darts_res_s5_3_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_6
```

### Trained model
| Model | Top-1 Accuracy | Download | MD5 |  
| :----:  | :--: | :--:  | :--:  |
|GM + DARTS (1st) | 97.65% | [[Google Drive]](https://drive.google.com/file/d/1qKbg65IlFWs75KIgJQuqNzHE9godZ435/view?usp=sharing)  [[Baidu Pan (pin:jsue)]](https://pan.baidu.com/s/1AWGIRIRrw9R7GxslkMMJGw)| 88ff04b2afbca067cf75c7430f776b96 |  

You can evaluate the trained model (GM + DARTS (1st)) with the following command:
```shell
bash eval.sh --arch search_darts_sota_ws_res_s5_2_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_5 --initial_checkpoint ../../checkpoint/search_darts_sota_ws_res_s5_2_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_5.pth.tar
```

### Successive halving
For detail of successive halving, please refer to successive_halving.py and use the following command:
```shell
python successive_halving.py exp_name exp_type (fs or ws)
```



