## layzy set up

import argparse
from fit_data import fit_model
from train_model import train_model
from eval_model import evaluate_model

def get_args_parser(parents = []):
    parser = argparse.ArgumentParser('Singleto3D', add_help=False, parents=parents)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--log_freq', default=1000, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=4096, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save_freq', default=200, type=int) 
    parser.add_argument('--device', default='cuda:0', type = str)
    parser.add_argument('--surfix', default='', type = str)
    parser.add_argument('--save', default='', type = str)
    parser.add_argument('--load_checkpoint', action='store_true')         
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--vis_training', action='store_true')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--q11",  action='store_true')
    parser.add_argument("--q12",  action='store_true')
    parser.add_argument("--q13",  action='store_true')
    parser.add_argument("--q21",  action='store_true')
    parser.add_argument("--q22",  action='store_true')
    parser.add_argument("--q23",  action='store_true')
    parser.add_argument("--q24",  action='store_true')
    parser.add_argument("--q25",  action='store_true')
    parser.add_argument("--q26",  action='store_true')
    args = parser.parse_args()
    print(args)

    if args.q11:
        args.type = 'vox'
        fit_model(args)

    if args.q12:
        args.type = 'mesh'
        fit_model(args)

    if args.q12:
        args.type = 'point'
        fit_model(args)

    if args.q21:
        args.type = 'vox'
        train_model(args)

    if args.q22:
        args.type = 'point'
        train_model(args)

    if args.q23:
        args.type = 'mesh'
        train_model(args)

    if args.q24:
        print("Specify the type of the model")
        evaluate_model(args)

    if args.q25:
        print("Specify the number of n_points, default will be 4096")
        args.type = 'point'
        train_model(args)