import argparse
import torch
from prune import LMDP,FMDP


def parse_int_list(value):
    # This function will take a string and convert it into a list of integers
    try:
        # Split the string by comma and convert each item to an integer
        return [int(i) for i in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List values must be integers separated by commas (e.g., 500,501,502)")

def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    group = parser.add_mutually_exclusive_group(required=True)  # This makes it required to choose one
    group.add_argument('--lmdp', action='store_true', help='Enable LMDP option')
    group.add_argument('--fmdp', action='store_true', help='Enable FMDP option')

    parser.add_argument('--cuda', action='store_true', help='Force to use CUDA if available')
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--retain_class_nums',type=parse_int_list, default=[50,70,84],help='List of class numbers to retain, separated by commas (e.g., 500,501,502)')
    parser.add_argument("--image_size",type=int,default=128)
    parser.add_argument('--reprune', action='store_true')

    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(args)

    if args.lmdp:
        pruner = LMDP(args)
    else:
        pruner = FMDP(args)
    pruner.prune()


