import argparse
import torch
import os
import datetime
from pipeline import Pipeline


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cuda', action='store_true', help='Force to use CUDA if available')
    parser.add_argument('--resnet', action='store_true')
    # parser.add_argument('--gen_embed',action='store_true')
    parser.add_argument("--model", type=str,required=True, help="loads pretrained MAE model")
    parser.add_argument('--n_epochs',type=int, default =20)
    parser.add_argument('--batch_size',type=int, default=64)

    args = parser.parse_args()
    return args

    


if __name__ == "__main__":
    args = args_parser()
    
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    print(args)

    # checkpoint_dir_base = './MAE'
        
    # # creating checkpoint and run_id based on date dand time 
    # args.run_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[-7:]
    # checkpoint_dir = os.path.join(checkpoint_dir_base, args.run_id)
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # args.checkpoint_dir = checkpoint_dir

    # create loaders, implement transforms
    pipeline = Pipeline(args)

    print(f'Initialized Pipeline')

    