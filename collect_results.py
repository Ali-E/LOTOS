import os
import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--attack_details', default='trans_e0.04_s') ## from the trans data collected every 10 epoch during the training 
parser.add_argument('--second', action='store_true')
parser.add_argument('--res_type', default='trans')
args = parser.parse_args()


if __name__ == '__main__':
    seed_list = [10**i for i in range(5)]

    if args.res_type == 'ensemble':
        args.attack_details = "last_pgd_0.04_0.1.csv"
        # args.attack_details = "exp_pgd_0.04_0.1.csv"
    
    elif args.res_type == 'blackbox':
        args.attack_details = "blackbox_exp_pgd_0.04_0.1.csv"
        # args.attack_details = "blackbox_last_pgd_0.04_0.1.csv"

    single_df_list = []
    single_df = None
    for seed in seed_list:
        try:
            if args.res_type == 'blackbox':
                single_result_file = args.base_classifier + str(int(seed)) + '_' + args.attack_details
            elif args.res_type == 'ensemble':
                single_result_file = args.base_classifier + str(int(seed)) + '/' + args.attack_details
            else:
                single_result_file = args.base_classifier + str(int(seed)) + '/' + args.attack_details + str(int(seed)) + '.csv' 
            print(single_result_file)

            single_df = pd.read_csv(single_result_file)
            if args.res_type == 'blackbox':
                single_df_list.append(single_df)
            else:
                single_df_list.append(single_df.values.squeeze())

        except Exception as e:
            print(e)
            continue

    if args.res_type == 'ensemble':
        columns = single_df.columns
        print(single_df_list)
        print('mean: ', np.mean(single_df_list, axis=0))
        print('std: ', np.std(single_df_list, axis=0))
        df = pd.DataFrame(np.array(single_df_list), columns=columns)
        df.to_csv(args.base_classifier + args.attack_details + '_avg.csv', index=False)

    elif args.res_type == 'blackbox':
        single_df = pd.concat(single_df_list, axis=0)
        print(single_df)
        print('mean:')
        print(single_df.mean(axis=0))
        print('std:')
        print(single_df.std(axis=0))

    else:
        avg_mat = np.mean(single_df_list, axis=0)
        avg_df = pd.DataFrame(avg_mat, columns=single_df.columns)
        avg_df.to_csv(args.base_classifier + args.attack_details + '_avg.csv', index=False)
            
