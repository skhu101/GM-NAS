### Requirements
* python packages
    * Python >= 3.6
    * Pytorch >= 1.5
    * ImageNet Dataset

* Some codes are borrowed from **PyTorch Image Models** ([https://github.com/rwightman/pytorch-image-models]) 


## GM + ProxylessNAS

### evaluate the searched model in GM + ProxylessNAS
```shell
./distributed_train.sh 8 data/ILSVRC/Data/CLS-LOC/ --model GM_ProxylessNAS_split_2_edge_2_3_group_subnetwork_1 --opt rmsproptf --opt-eps .001 --output GM_ProxylessNAS_split_2_edge_2_3_group_subnetwork_1
```

### Trained model
| Model | Top-1 Test Error | Top-5 Test Error | Download | MD5 |  
| :----:  | :--: | :--:  | :--:  | :--:  |
|GM + ProxylessNAS | 23.4% | 7.0% | [[Google Drive]](https://drive.google.com/file/d/17azGLyfcCCP0IfGVoDpBVaPAqbdVyl42/view?usp=sharing)  [[Baidu Pan (pin:3cn4)]](https://pan.baidu.com/s/1tWEQu206Z3_ZW7-I12ohfw)| c861e884b0007baa71e815f7a301d4f3 |  

You can evaluate the trained model (GM + ProxylessNAS) with the following command:
```shell
./distributed_train.sh 8 /home/yangkuo/data/ImageNet/imagenet_ilsvrc/ILSVRC/Data/CLS-LOC/ --model GM_ProxylessNAS_split_2_edge_2_3_group_subnetwork_1 --opt rmsproptf --opt-eps .001 --output GM_ProxylessNAS_split_2_edge_2_3_group_subnetwork_1 --initial-checkpoint checkpoint/GM_ProxylessNAS_split_2_edge_2_3_group_subnetwork_1.pth.tar
```


## GM + OFA

### evaluate the searched model in GM + OFA
Download the teacher model from [[Google Drive]](https://drive.google.com/file/d/1ZMPntl17VU2RJhhdvOgkCpY2tlb21_eR/view?usp=sharing) or [[Baidu Pan (pin:usd1)]](https://pan.baidu.com/s/1nZQ5sic1NjtSovSe-AguzQ) MD5 (22aea7045b519392431638ccadfdb278) and put the teacher model in timm/models/my_models
```shell
./distributed_train.sh 8 data/ILSVRC/Data/CLS-LOC/ --model GM_OFA_split_3_edge_2_group_subnetwork_1_flops_584 --opt lookahead_rmsproptf --opt-eps .001  --knowledge_distill --kd_ratio 9.0 --teacher_name D-Net-big224 --output GM_OFA_split_3_edge_2_group_subnetwork_1_flops_584
```

### Trained model
| Model | Top-1 Test Error | Top-5 Test Error | Download | MD5 |  
| :----:  | :--: | :--:  | :--:  | :--:  |
|GM + OFA | 19.4% | 4.9% | [[Google Drive]](https://drive.google.com/file/d/15pSUdP2ko5Av9WciWH7dO6m6h2n0bFHs/view?usp=sharing)  [[Baidu Pan (pin:mo65)]](https://pan.baidu.com/s/1dXKwsO4OCrMDwXE5jKl_Gw)| ab14b8e0255a2550ad5b1a674531275c |  

You can evaluate the trained model (GM + OFA) with the following command:
```shell
./distributed_train.sh 8 data/ILSVRC/Data/CLS-LOC/ --model GM_OFA_split_3_edge_2_group_subnetwork_1_flops_584 --opt lookahead_rmsproptf --opt-eps .001  --knowledge_distill --kd_ratio 9.0 --teacher_name D-Net-big224 --output GM_OFA_split_3_edge_2_group_subnetwork_1_flops_584 --initial-checkpoint checkpoint/GM_OFA_split_3_edge_2_group_subnetwork_1_flops_584.pth.tar
```

