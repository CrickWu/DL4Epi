#!/usr/bin/env python
# encoding: utf-8

import glob
import numpy as np

## Find the best performance in sets of logs
# extract the values from log
def extract_tst_from_log(filename):
    # empty file
    lines = open(filename).readlines()
    if len(lines) < 1:
        return 1e10, -1
    line = lines[-1]
    # invalid or NaN
    if not line.startswith('test rse'):
        return 1e10, -1
    fields = line.split('|')
    tst_rse = float(fields[0].split()[2])
    tst_cor = float(fields[2].split()[2])
    return tst_rse, tst_cor


def format_logs(raw_expression):
    val_filenames = []
    for num in [1, 2, 4, 8]:
        expressions = raw_expression.format(num)
        filenames = glob.glob(expressions)
        tuple_list = [extract_tst_from_log(filename) for filename in filenames]
        if len(tuple_list) == 0:
            continue
        rse_list, cor_list = zip(*tuple_list)
        index = np.argmin(rse_list)
        print 'horizon:{:2d}'.format(num), 'rmse: {:.4f}'.format(rse_list[index]), 'corr: {:.4f}'.format(cor_list[index]), 'best_model:', filenames[index]

if __name__ == '__main__':
    for data in ['hhs', 'regsions']:
        print 'Dataset:', data
        # cnnrnn_res
        print '*' * 40
        format_logs('./log/cnnrnn_res/cnnrnn_res.%s.hid-*.drop-*.w-*.h-{}.ratio-*.res-*.out' % (data))
        print '*' * 40
        # cnnrnn
        format_logs('./log/cnnrnn/cnnrnn.%s.hid-*.drop-*.w-*.h-{}.out' %(data))
        print '*' * 40
        # rnn
        format_logs('./log/rnn/rnn.%s.hid-*.drop-*.w-*.h-{}.out' %(data))
        print '*' * 40
        # gar
        format_logs('./log/gar/gar.%s.d-*.w-*.h-{}.out' %(data))
        print '*' * 40
        # ar
        format_logs('./log/ar/ar.%s.d-*.w-*.h-{}.out' %(data))
        print '*' * 40
        # var
        format_logs('./log/var/var.%s.d-*.w-*.h-{}.out' %(data))
        print '*' * 40