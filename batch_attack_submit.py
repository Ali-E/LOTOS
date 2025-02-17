import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--bc2", default='', type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--attack', default='pgd')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--num_models', default=3, type=int)
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--choice', default='exp')
parser.add_argument('--cat', default=0, type=int)
parser.add_argument('--cross', default=0, type=int)
parser.add_argument('--single', default=1, type=int)
parser.add_argument('--blackbox', default=0, type=int)
parser.add_argument('--fBN', default=0, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    base_classifier = args.base_classifier
    base_classifier_2 = args.bc2
    dataset = args.dataset
    attack = args.attack
    seed = args.seed    
    model = args.model

    if seed == -1:
        seed_list = [10**i for i in range(3)]
    elif seed == -2:
        seed_list = [10**i for i in range(2)]
    else:
        seed_list = [seed]

    steps = 50

    method = args.method
    if method == 'all':
        methods = ['orig', 'fastclip_tlower_cs100']
    elif method == 'clip':
        methods = ['fastclip_cs100']
    elif method[:4] == 'fast':
        methods = ['fastclip_tlower_cs100', 'fastclip_tlower_cs50']
    else:
        methods = [method]

    if args.model  not in ['ResNet18', 'ResNet34', 'DLA', 'SimpleConv', 'VGG19']:
        raise ValueError('model must be one of ResNet18, DLA, SimpleConv')

    mode = args.mode
    if mode == 'all':
        modes = ['wBN', 'noBN']
    else:
        modes = [mode]

    choice = args.choice


    for mode in modes:
        for method in methods:
            if method == 'orig' and (mode == 'clipBN_hard'):
                continue
            for seed in seed_list:
                try:
                    if args.cross == 1:
                        if args.blackbox == 1:
                            command = f"sbatch attack_blackbox_submit.slurm  {base_classifier} {base_classifier_2} {method} {mode} {attack} {choice} {seed} {args.num_models} {args.dataset} {args.model}"
                        elif args.single == 1:
                            command = f"sbatch attack_ens_single_submit.slurm  {base_classifier} {base_classifier_2} {method} {mode} {attack} {choice} {seed} {args.num_models}"
                        else:
                            command = f"sbatch attack_ens_submit.slurm  {base_classifier} {base_classifier_2} {method} {mode} {attack} {choice} {seed} {args.num_models}"

                        if args.cat == 1:
                            command += ' --cat'
                    else:
                        if args.cat == 1:
                            if args.fBN == 1:
                                command = f"sbatch attack_job_submit.slurm  {base_classifier} {args.model} {method} {mode} {attack} {choice} {seed} {args.num_models} --cat --fBN"
                            else:
                                command = f"sbatch attack_job_submit.slurm  {base_classifier} {args.model} {method} {mode} {attack} {choice} {seed} {args.num_models} --cat"
                        else:
                            if args.fBN == 1:
                                command = f"sbatch attack_job_submit.slurm  {base_classifier} {args.model} {method} {mode} {attack} {choice} {seed} {args.num_models} --fBN"
                            else:
                                command = f"sbatch attack_job_submit.slurm  {base_classifier} {args.model} {method} {mode} {attack} {choice} {seed} {args.num_models}"

                    print(command)
                    os.system(command)

                except Exception as e:
                    print(e)
                    continue
