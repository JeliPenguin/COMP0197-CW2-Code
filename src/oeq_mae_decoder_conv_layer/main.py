import argparse
import torch
from src.oeq_mae_decoder_conv_layer.pipeline import Pipeline


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_size',type=int, default=128)
    parser.add_argument('--patch_size',type=int, default=4)

    parser.add_argument('--encoder_width',type=int, default=1024)
    parser.add_argument('--n_heads',type=int, default=16)
    parser.add_argument('--encoder_depth',type=int, default=12)
    parser.add_argument('--decoder_width',type=int, default=512)
    parser.add_argument('--decoder_depth',type=int, default=12)
    parser.add_argument('--mlp_ratio',type=float, default=4.0)
    parser.add_argument('--dropout',type=float, default=0.1)
    parser.add_argument('--mask_ratio',type=int, default=0.8)
    parser.add_argument('--no_cls_token_encoder',action='store_true')
    parser.add_argument('--no_cls_token_decoder',action='store_true')
    parser.add_argument("--model", type=str, default="./models/pretrained_encoder", help="loads pretrained MAE model")
    parser.add_argument('--checkpoint_dir',type=str, default="./models/trained_models")
    parser.add_argument('--c',type=int, default=3)
    parser.add_argument('--cuda', action='store_true', help='Force to use CUDA if available')
    
    parser.add_argument('--output_display_period',type=int, default=10)
    parser.add_argument('--batch_print_period',type=int, default=5)
    parser.add_argument('--batch_size',type=int, default=4)
    parser.add_argument('--out_indices',type=list, default=[1, 3, 5, 7, 11])
    

    parser.add_argument('--refinetune',action='store_true')
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = args_parser()
    
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    pipeline = Pipeline(args)

    print(f'Initialized Pipeline')

    if args.refinetune:
        print("TRAIN ENCODER AND DECODER 25 EPOCH")
        pipeline.train(n_epoch=25, freeze_encoder=False, freeze_decoder=False)
        print("TRAIN ENCODER 25 EPOCH")
        pipeline.train(n_epoch=25, freeze_encoder=False, freeze_decoder=True)
        print("TRAIN DECODER 25 EPOCH")
        pipeline.train(n_epoch=25, freeze_encoder=True, freeze_decoder=False)
    else:
        pipeline.load_model_checkpoint()
    
    print("Testing...")

    pipeline.test()
    pipeline.show_examples()

    