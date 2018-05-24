#!/bin/bash

window_list=(2 8 32 64 128)
horizon_list=(1 2 4 8)
hid_list=(5 10 20 40)
dropout_list=(0.0 0.2 0.5)
# residual window size
res_list=(4 8 16)
ratio_list=(0.001 0.01 0.1 1)


DATA=$1
SIM_MAT=$2
LOG=$3
GPU=$4
NORM=$5

for horizon in "${horizon_list[@]}"
do
  for ratio in "${ratio_list[@]}"; do
    for window in "${window_list[@]}"; do
    for dropout in "${dropout_list[@]}"; do
    for hid in "${hid_list[@]}"; do
    for res in "${res_list[@]}"; do
	    cnn_option="--sim_mat ${SIM_MAT}"
        rnn_option="--hidRNN ${hid} --residual_window ${res}"
        option="--ratio ${ratio} --dropout ${dropout} --normalize ${NORM} --epochs 2000 --data ${DATA} --model CNNRNN_Res --save_dir save --save_name cnnrnn_res.${LOG}.w-${window}.h-${horizon}.res-${res}.pt --horizon ${horizon} --window ${window} --gpu ${GPU} --metric 0"
        cmd="stdbuf -o L python ./main.py ${option} ${cnn_option} ${rnn_option} | tee log/cnnrnn_res/cnnrnn_res.${LOG}.hid-${hid}.drop-${dropout}.w-${window}.h-${horizon}.ratio-${ratio}.res-${res}.out"
        echo $cmd
        eval $cmd
    done
    done
    done
    done
  done
done
