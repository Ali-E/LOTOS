import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def compute_trans(sub_df, perturb_robust=False, model_count=3):
    if model_count == 3:
        trans_mat = sub_df[['t0', 't1', 't2']].values
    elif model_count == 2:
        trans_mat = sub_df[['t0', 't1']].values
    acc_avg = sub_df['acc'].mean()

    if perturb_robust:
        perturb_robustness = sub_df['perturb_robust'].mean()

    # compute the sum of off-diagonal values of trans_mat:
    off_diagonal_sum = np.sum(trans_mat) - np.trace(trans_mat)
    if model_count == 3:
        trans_rate = off_diagonal_sum / 6.
    elif model_count == 2:
        trans_rate = off_diagonal_sum / 2.

    if model_count == 3:
        robustness = 1. - np.trace(trans_mat) / 3.
    elif model_count == 2:
        robustness = 1. - np.trace(trans_mat) / 2.

    if perturb_robust:
        return trans_rate, robustness, acc_avg, perturb_robustness

    return trans_rate, robustness, acc_avg


if __name__ == '__main__':

    perturb_robust = False 
    model_count = 3

    trans_file_1 = sys.argv[1]
    trans_df_1 = pd.read_csv(trans_file_1)
    print(trans_df_1.head())

    # group by epoch values:
    trans_df_1_group = trans_df_1.groupby('epoch')
    results = []
    for idx, group in trans_df_1_group:
        print(idx)
        group = group[:model_count]
        if perturb_robust:
            trans_rate, robustness, acc_avg, pr = compute_trans(group, perturb_robust=perturb_robust)
            results.append([int(idx),trans_rate, robustness, acc_avg, pr])
        else:
            trans_rate, robustness, acc_avg = compute_trans(group, perturb_robust=perturb_robust)
            results.append([int(idx),trans_rate, robustness, acc_avg])

    if perturb_robust:
        results = pd.DataFrame(results, columns=['epoch', 'trans', 'robustness', 'acc', 'perturb_robustness'])
    else:
        results = pd.DataFrame(results, columns=['epoch', 'trans', 'robustness', 'acc'])
    results.to_csv(trans_file_1[:-4] + '_summary.csv', index=False)