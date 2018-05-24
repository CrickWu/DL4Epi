#!/bin/bash

window_list=(2 8 32 64 128)
horizon_list=(1 2 4 8)
decay_list=(0.01 0.1 1)

DATA=$1
LOG=$2
GPU=$3
NORM=$4

for window in "${window_list[@]}"
do
    for HORIZON in "${horizon_list[@]}"
    do
    	for decay in "${decay_list[@]}"
        do
            option="--normalize ${NORM} --epochs 1500 --save_dir save --save_name gar.${LOG}.w-${window}.h-${HORIZON}.pt --data ${DATA} --model GAR --horizon ${HORIZON} --window ${window} --gpu ${GPU} --metric 0 --weight_decay ${decay}"
            cmd="stdbuf -o L python ./main.py ${option} | tee log/gar/gar.${LOG}.d-${decay}.w-${window}.h-${HORIZON}.out"
            echo $cmd
            eval $cmd
        done
    done
done
