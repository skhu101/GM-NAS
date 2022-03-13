# take sub-supernet 1 for example
declare -a tasks=("kernel" "depth" "expand")
for task in "${tasks[@]}"
do
	if [ "$task" = "kernel" ] || [ "$task" = "normal" ];
	then
	    python -m torch.distributed.launch --nproc_per_node=8 train_ofa_net_ws_search.py --task=$task --initial_enc exp/GM_split_3_edge_2_group_subnetwork_1_kernel_size_enc.txt
	    echo "$task is executed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	else
	    declare -a phases=(1 2)
	    for phase in "${phases[@]}"
	    do
	        python -m torch.distributed.launch --nproc_per_node=8 train_ofa_net_ws_search.py --task=$task --phase=$phase --initial_enc exp/GM_split_3_edge_2_group_subnetwork_1_kernel_size_enc.txt
	    done
	fi
done

mv exp GM_split_3_edge_2_group_subnetwork_1
