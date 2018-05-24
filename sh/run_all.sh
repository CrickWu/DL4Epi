#!/bin/bash

# us_hhs experiments
bash ./sh/grid_AR.sh ./data/us_hhs/data.txt hhs 0 1
bash ./sh/grid_VAR.sh ./data/us_hhs/data.txt hhs 0 1
bash ./sh/grid_GAR.sh ./data/us_hhs/data.txt hhs 0 1
bash ./sh/grid_RNN.sh ./data/us_hhs/data.txt hhs 0 1
bash ./sh/grid_CNNRNN.sh ./data/us_hhs/data.txt ./data/us_hhs/ind_mat.txt hhs 0 1
bash ./sh/grid_CNNRNN_Res.sh ./data/us_hhs/data.txt ./data/us_hhs/ind_mat.txt hhs 0 1

# us_regions experiments
bash ./sh/grid_AR.sh ./data/us_regions/data.txt regions 0 1
bash ./sh/grid_VAR.sh ./data/us_regions/data.txt regions 0 1
bash ./sh/grid_GAR.sh ./data/us_regions/data.txt regions 0 1
bash ./sh/grid_RNN.sh ./data/us_regions/data.txt regions 0 1
bash ./sh/grid_CNNRNN.sh ./data/us_regions/data.txt ./data/us_regions/ind_mat.txt regions 0 1
bash ./sh/grid_CNNRNN_Res.sh ./data/us_regions/data.txt ./data/us_regions/ind_mat.txt regions 0 1
