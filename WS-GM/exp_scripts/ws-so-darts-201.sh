#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-1}
gpu=${gpu:-"auto"}
group=${group:-"none"}

## dev
expid_tag=${expid_tag:-"none"}
dis_metric=${dis_metric:-"cos"}
split_eids=${split_eids:-"none"}
edge_crit=${edge_crit:-"predefine"}

## hyper
split_ckpts=${split_ckpts:-'10,20'}
projection_warmup_epoch=${projection_warmup_epoch:-20}
space=${space:-""}

## restart
restart=${restart:-"none"}
warmup=${warmup:-0}

## hard/soft grad
hard=${hard:-1}

## align with few-shot optimization
fs_opt=${fs_opt:-0}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset 'gpu:' $gpu
echo 'dis_metric:' $dis_metric 'split_eids:' $split_eids

cd ../nasbench201/
python train_search.py \
    --method ws-so --supernet_train darts --projection darts \
    --search_space nas-bench-201${space} \
    --dataset $dataset \
    --group $group --save $id --gpu $gpu --seed $seed \
    --split_eids $split_eids --split_crit 'grad' --split_num 2 --dis_metric $dis_metric \
    --split_ckpts $split_ckpts --projection_warmup_epoch $projection_warmup_epoch \
    --edge_crit $edge_crit \
    --restart $restart --warmup $warmup \
    --hard $hard \
    --fs_opt $fs_opt \