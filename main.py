import argparse
import torch
from pipeline import Pipeline


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cuda', action='store_true', help='Force to use CUDA if available')
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--facebook_mae', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    # parser.add_argument('--gen_embed',action='store_true')
    parser.add_argument("--model", type=str,required=True, help="loads pretrained MAE model")
    parser.add_argument('--n_epochs',type=int, default =20)
    parser.add_argument('--batch_size',type=int, default=16)

    args = parser.parse_args()
    return args

    


if __name__ == "__main__":
    args = args_parser()
    
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    print(args)

    pipeline = Pipeline(args)

    print(f'Initialized Pipeline')

    if not args.test_mode:
        pipeline.train()

    
    pipeline.test()
    pipeline.show_examples()

    