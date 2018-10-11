# Deep Learning for Epidemiological Predictions

## Paper
[Deep Learning for Epidemiological Predictions](https://raw.githubusercontent.com/CrickWu/crickwu.github.io/master/papers/sigir2018.pdf) - Yuexin Wu et. al. SIGIR 2018

The overall structure is composed of 3 parts: a CNN for capturing correlation between signals, a RNN for linking up the dependencies in the temporal dimension and the residual links for fast training and overfitting prevention. We carefully restrain the parameter space, making the total model have a similar size as autogression models.

<p style="text-align:center;"><img src="https://github.com/CrickWu/DL4Epi/blob/master/figs/framework.png" alt="framework" style="max-width:80%;"></p>


## Dependencies
```Python == 2.7, Pytorch == 0.2.0, numpy```
## How to Run
Preprocessing: run `bash ./sh/mklog.sh` to create empty `log` `save` folders.
### Simple Example
```
python main.py --normalize 1 --epochs 2000 --data ./data/us_hhs/data.txt --sim_mat ./data/us_hhs/ind_mat.txt --model CNNRNN_Res \
--dropout 0.5 --ratio 0.01 --residual_window 4 --save_dir save --save_name cnnrnn_res.hhs.w-16.h-1.ratio.0.01.hw-4.pt \
--horizon 1 --window 16 --gpu 0 --metric 0
```
### Experiment for a Single Dataset/Method
For CNNRNN_Res, CNNRNN:
```
bash ./sh/grid_<model>.sh <data_path> <adj_mat_path> <log_info> <gpu_number> <normalization>
```
e.g.
```
bash ./sh/grid_CNNRNN_Res.sh ./data/us_hhs/data.txt ./data/us_hhs/ind_mat.txt hhs 0 1
```
For VAR, GAR, AR:
```
bash ./sh/grid_<model>.sh <data_path> <log_info> <gpu_number> <normalization>
```
e.g.
```
bash ./sh/grid_VAR.sh ./data/us_hhs/data.txt hhs 0 1
```

Use `python log_parse.py` to parse the results in `log\`.
### Full Experiment in Paper
***NOTICE: This may take LONG time. Consider running single experiments first to estimate time.***

```
bash ./sh/run_all.sh
python log_parse.py
```

## Option Explanation
For `main.py`

```
normalize: normalization options
	0: no normalization
	1: signal/row-wise normalization
	2: global/matrix-wise normalization
```
More information can be found in `main.py --help`.

## Log Format
```
rse: Root-mean-square error
rae: Absolute error
correlation: Pearson correlation score
```

## Notice
The results in the paper are produced by pytorch-0.2.0, which seems to have some unstable numerical issues and is likely to produce NaN in some cases. If that happens, you may try rerun the code or switch to a more stable pytorch version (e.g. 0.4.0) for more robust prediction.

## Citation
```
@inproceedings{wu2018deep,
  title={Deep Learning for Epidemiological Predictions},
  author={Wu, Yuexin and Yang, Yiming and Nishiura, Hiroshi and Saitoh, Masaya},
  booktitle={The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  pages={1085--1088},
  year={2018},
  organization={ACM}
}
```
