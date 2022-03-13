#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"auto"}

space=${space:-s5}

## dev
expid_tag=${expid_tag:-"none"}
dis_metric=${dis_metric:-"cos"}
split_eid=${split_eid:-0}
edge_crit=${edge_crit:-"rand"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset 'space:' $space
echo 'gpu:' $gpu

cd ../sota/cnn
python train_search_ws.py \
    --method ws-so --supernet_train darts --projection darts \
    --search_space $space --dataset $dataset \
    --seed $seed --save $id --gpu $gpu \
    --dis_metric $dis_metric --split_eid $split_eid --edge_crit $edge_crit --split_crit 'grad' --split_num 2 \
    --split_ckpts 2,4,6 --projection_warmup_epoch 15 \
    --fix_alpha_equal --restart 'final' \
    --fix_sche 1 \
    # --expid_tag debug --fast \