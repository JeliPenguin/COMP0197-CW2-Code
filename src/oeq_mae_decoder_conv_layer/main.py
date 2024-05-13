import argparse
import torch
from src.oeq_mae_decoder_conv_layer.pipeline import Pipeline


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cuda', action='store_true', help='Force to use CUDA if available')
    parser.add_argument("--model", type=str,required=True, help="loads pretrained MAE model")
    parser.add_argument('--batch_size',type=int, default=16)

    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = args_parser()
    
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    pipeline = Pipeline(args)

    print(f'Initialized Pipeline')

    print("TRAIN ENCODER AND DECODER 25 EPOCH")
    pipeline.train(n_epoch=25, freeze_encoder=False, freeze_decoder=False)
    print("TRAIN ENCODER 25 EPOCH")
    pipeline.train(n_epoch=25, freeze_encoder=False, freeze_decoder=True)
    print("TRAIN DECODER 25 EPOCH")
    pipeline.train(n_epoch=25, freeze_encoder=True, freeze_decoder=False)

    pipeline.test()
    pipeline.show_examples()

    