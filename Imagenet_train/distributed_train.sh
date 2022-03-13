#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port 41586 train.py -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --model-ema --model-ema-decay 0.9999 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 "$@"
