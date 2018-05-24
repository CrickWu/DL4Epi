#!/bin/bash

log=${1:-.}
mkdir ${log}

dirlist=(ar gar var rnn cnnrnn cnnrnn_res)
for subfolder in "${dirlist[@]}"; do
    mkdir ${log}/${subfolder}
done

mkdir ${log}/save
