### Requirements
* python packages
    * Python >= 3.7
    * PyTorch == 1.0

* Some codes are borrowed from **once-for-all** ([https://github.com/mit-han-lab/once-for-all]) 


### GM + OFA supernet partition
```shell
python -m torch.distributed.launch --nproc_per_node=8 train_ofa_net.py --task normal
python -m torch.distributed.launch --nproc_per_node=8 weight_sharing_train_ofa_net.py --exp_name exp/GM_grad_split_3_edge_2_group --task kernel --split_ckpts 30,60,90 --split_num '2,2,2' --edge_crit grad
```
Note that --split_ckpts refers to the epoch number to perform the split, --split_num refers to splitting the operations on the selected edge into $split_num groups


### GM + OFA sub-supernet search (take sub-supernet 1 for example)
After splitting, you need to write the encodings for each supernet by using the same format as that in exp/GM_split_3_edge_2_group_subnetwork_1_kernel_size_enc.txt and then using the following command to perform the search
```shell
bash ofa_supernet_search.sh
```

### Evolution search based on splitted sub-supernets (take sub-supernet 1 for example)
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 evolution_search.py --port 23346 --initial_enc exp/GM_split_3_edge_2_group_subnetwork_1_kernel_size_enc.txt --initial_model GM_split_3_edge_2_group_subnetwork_1/kernel_depth2kernel_depth_width/phase2/checkpoint/model_best.pth.tar
```

### generate the config with the configuration from evolution search
```shell
python generate_net_config.py
```

### Evaluate searched model
Please refer to Imagenet_train/README.md

