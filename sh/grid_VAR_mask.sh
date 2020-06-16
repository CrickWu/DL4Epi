#!/bin/bash

window_list=(2 8 32 64 128)
horizon_list=(1 2 4 8)
decay_list=(0.01 0.1 1)

DATA=$1
SIM_MAT=$2
LOG=$3
GPU=$4
NORM=$5

for window in "${window_list[@]}"
do
    for HORIZON in "${horizon_list[@]}"
    do
        for decay in "${decay_list[@]}"
        do
            option="--normalize ${NORM} --epochs 1000 --data ${DATA} --model VAR_mask --save_dir save --save_name var_mask.${LOG}.w-${window}.h-${HORIZON}.pt --horizon ${HORIZON} --window ${window} --gpu ${GPU} --metric 0 --weight_decay ${decay} --sim_mat ${SIM_MAT}"
            cmd="stdbuf -o L python ./main.py ${option} | tee log/var_mask/var_mask.${LOG}.d-${decay}.w-${window}.h-${HORIZON}.out"
            echo $cmd
            eval $cmd
        done
    done
done
