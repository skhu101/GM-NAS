### Requirements
* python packages
    * Python >= 3.6
    * Pytorch == 1.0
    * ImageNet Dataset

* Some codes are borrowed from **ProxylessNAS** ([https://github.com/mit-han-lab/proxylessnas]) 


### Search
GM + ProxylessNAS :
```shell
python search/weight_sharing_imagenet_arch_search.py --path experiments/GM_ProxylessNAS_split_2_edge_2_3_group --train_batch_size 512 --split_ckpts 40,80 --split_num 2,3 --edge_crit grad
```
Note that --split_ckpts refers to the epoch number to perform the split, --split_num refers to splitting the operations on the selected edge into $split_num groups

### Evaluate searched model
Please refer to Imagenet_train/README.md








