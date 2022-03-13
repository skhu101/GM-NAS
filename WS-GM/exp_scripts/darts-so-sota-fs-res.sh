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
split_num=${split_num:-7}

## hyper
split_ckpts=${split_ckpts:-'0'}
projection_warmup_epoch=${projection_warmup_epoch:-20}

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
    --dis_metric $dis_metric --split_eid $split_eid  --split_crit 'fewshot' --skip_final_split 1 --split_num $split_num \
    --split_ckpts $split_ckpts --projection_warmup_epoch $projection_warmup_epoch \
    --fix_alpha_equal --restart 'final' \
    --fix_sche 1 \
#    --expid_tag debug --fast \