import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--job_type', default='train', help='type of job to run (train-evaluate)')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--num_models', default=3, type=int)
parser.add_argument('--mode', default='wBN')
parser.add_argument('--seed', default=-1, type=int)
parser.add_argument('--conv_factor', default=0.00, type=float)
parser.add_argument('--cat_factor', default=0.00, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--bottom_clip', default=1.0, type=float)
parser.add_argument('--cat_bottom_clip', default=1.0, type=float)
parser.add_argument('--widen_factor', default=1, type=int, help='widen factor for WideResNet')
parser.add_argument('--cat', default=0, type=int)
parser.add_argument('--fBN', default=0, type=int)
parser.add_argument('--fOrtho', default=0, type=int)
parser.add_argument('--efe', default=0, type=int)
parser.add_argument('--tech', default='trsl2', help='the prior method to use')
parser.add_argument('--dir', default='bblr', help='the prior method to use')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.dataset
    model = args.model
    if args.seed == -1:
        seed_list = [10**i for i in range(5)]
    elif args.seed == -2:
        seed_list = [10**i for i in range(2)]
    else:
        seed_list = [args.seed]

    convsn_list = [1.]

    if args.bottom_clip == -1:
        bottom_list = [0.7,0.8, 0.9]
    else:
        bottom_list = [args.bottom_clip]

    if args.cat_bottom_clip == -1:
        cat_bottom_list = [0.8, 1.0]
    else:
        cat_bottom_list = [args.cat_bottom_clip]

    if args.conv_factor == -1:
        conv_factor_list = [0.01,0.03,0.05]
    else:
        conv_factor_list = [args.conv_factor]

    if args.cat_factor == -1:
        cat_factor_list = [0.01,0.03,0.05]
    else:
        cat_factor_list = [args.cat_factor]


    steps = 50 # this is clipBN steps

    method = args.method
    if method == 'all':
        methods = ['orig', 'fastclip_tlower_cs100']
    elif method == 'clip':
        methods = ['fastclip_cs100']
    elif method[:4] == 'fast':
        methods = ['fastclip_tlower_cs100']
    else:
        methods = [method]

    if args.model  not in ['ResNet18', 'ResNet34', 'DLA']:
        raise ValueError('model must be one of ResNet18, DLA, ResNet34')

    mode = args.mode
    if mode == 'all':
        modes = ['wBN', 'noBN']
    else:
        modes = [mode]

    for mode in modes:
        for method in methods:
            if method == 'orig':
                convsn_list_tmp = [1.0]
            else:
                convsn_list_tmp = convsn_list
            for seed in seed_list:
                for convsn in convsn_list_tmp:
                    for bottom in bottom_list:
                        for cat_bottom in cat_bottom_list:
                            for conv_factor in conv_factor_list:
                                for cat_factor in cat_factor_list:
                                    try:
                                        if args.job_type == 'train':
                                            command = f"sbatch job_submit.slurm {method} {mode} {seed} {convsn} {conv_factor} {cat_factor} {bottom} {cat_bottom} {args.widen_factor} {args.model} {args.lr} {args.dataset} {args.efe} {args.num_models} {args.tech} {args.dir}"

                                        if args.cat == 1:
                                            if args.fBN == 1:
                                                command += " --cat --fBN"
                                            else:
                                                command += " --cat"
                                        else:
                                            if args.fBN == 1:
                                                command += " --fBN"

                                        if args.fOrtho == 1:
                                            command += " --fOrtho"

                                        print(command)
                                        os.system(command)

                                    except Exception as e:
                                        print(e)
                                        continue

