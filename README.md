# GM-NAS

This repository contains the PyTorch implementation of the paper:<br/>
[**Generalizing Few-Shot NAS with Gradient Matching**](https://openreview.net/pdf?id=_jMtny3sMKU) in *ICLR 2022*.

By Shoukang Hu*, Ruochen Wang*, Lanqing Hong, Zhenguo Li, Cho-Jui Hsieh, and Jiashi Feng.<br/><br/>


<p align="left">
    <img src="img/GM_NAS.png" height="300"/>
</p>


For experiments on **NASBench-201 and DARTS space**, please refer to **WS-GM/README.md**

For experiments on **ProxylessNAS space**, please refer to **ProxylessNAS-GM/README.md**

For experiments on **OFA space**, please refer to **once-for-all-GM/README.md**

For **evaluating searched architectures** from ProxylessNAS and OFA space, please refer to **Imagenet_train/README.md**



### Patch Note (Oct 30, 2022)
There has been a logging error in NB201's architecture selection phase that causes some confusion in reproducibility. We've updated the logging. For more details on the architecture selection method, please refer to Appendix C of the paper.


### Citation
If you find our codes or trained models useful in your research, please consider to star our repo and cite our paper:

    @inproceedings{hu2022generalizing,
      title={Generalizing Few-Shot NAS with Gradient Matching},
      author={Hu, Shoukang and Wang, Ruochen and Lanqing, HONG and Li, Zhenguo and Hsieh, Cho-Jui and Feng, Jiashi},
      booktitle={International Conference on Learning Representations},
      year={2022}
    }

