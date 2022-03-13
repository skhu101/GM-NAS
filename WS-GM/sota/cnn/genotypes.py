from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'noise',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# search-darts-sota-ws-res-s5-0_[cos]_[e0]_[edge_crit_grad]_[split_crit_grad]-fix_alpha_equal_[res-final]_[fix_sche]_[[2, 4, 6]|15]
search_darts_sota_ws_res_s5_0_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_1=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# search-darts-sota-ws-res-s5-1_[cos]_[e0]_[edge_crit_grad]_[split_crit_grad]-fix_alpha_equal_[res-final]_[fix_sche]_[[2, 4, 6]|15]
search_darts_sota_ws_res_s5_1_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_0=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))

# search-darts-sota-ws-res-s5-2_[cos]_[e0]_[edge_crit_grad]_[split_crit_grad]-fix_alpha_equal_[res-final]_[fix_sche]_[[2, 4, 6]|15]
search_darts_sota_ws_res_s5_2_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_5=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 1), ('dil_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# search-darts-sota-ws-res-s5-3_[cos]_[e0]_[edge_crit_grad]_[split_crit_grad]-fix_alpha_equal_[res-final]_[fix_sche]_[[2, 4, 6]|15]
search_darts_sota_ws_res_s5_3_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_6=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

# search-darts-so-sota-ws-res-s5-0_[cos]_[e0]_[edge_crit_grad]_[split_crit_grad]-fix_alpha_equal_[res-final]_[fix_sche]_[[2, 4, 6]|15]
search_darts_so_sota_ws_s5_0_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_7=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

#search-darts-so-sota-ws-s5-1_cos_e0_edge_crit_grad_split_crit_grad_lr_decay-fix_alpha_equal-model_reinit_after_split-split_ckpts-2_4_5_projection_warmup_epoch_15
search_darts_so_sota_ws_s5_1_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_4=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))

#search-darts-so-sota-ws-s5-2_cos_e0_edge_crit_grad_split_crit_grad_lr_decay-fix_alpha_equal-model_reinit_after_split-split_ckpts-2_4_5_projection_warmup_epoch_15
search_darts_so_sota_ws_s5_2_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_6=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

#search-darts-so-sota-ws-s5-3_cos_e0_edge_crit_grad_split_crit_grad_lr_decay-fix_alpha_equal-model_reinit_after_split-split_ckpts-2_4_5_projection_warmup_epoch_15
search_darts_so_sota_ws_s5_3_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_0=Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

# search-snas-ws-darts-res-s5-0_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res-final_fix_sche_2_4_6_15
search_snas_ws_darts_res_s5_0_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_6=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

# search-snas-ws-darts-res-s5-1_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res-final_fix_sche_2_4_6_15
search_snas_ws_darts_res_s5_1_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_6=Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 2), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

# search-snas-ws-darts-res-s5-2_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res-final_fix_sche_2_4_6_15
search_snas_ws_darts_res_s5_2_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_2=Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# search-snas-ws-darts-res-s5-3_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res-final_fix_sche_2_4_6_15
search_snas_ws_darts_res_s5_3_cos_e0_edge_crit_grad_split_crit_grad_fix_alpha_equal_res_final_fix_sche_2_4_6_15_id_6=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))

# few shot nas architecture
few_shot_NAS_arch = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0),('skip_connect', 2), ('skip_connect', 3)], reduce_concat=[2, 3, 4,5])
