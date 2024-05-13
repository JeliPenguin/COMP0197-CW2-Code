import torch
import datetime
from train_mae import Trainer
from src.mae_code.mae_utils import save_config
import os
import argparse


# three modes:
    #pretrain just pretrains an encoder using masked auto encoder approach 
    # for image segmentation

    # finetune: specify an  --encoder path to add a decoder to an already finetuned encoder at encoder path; if none given, the model's will be pretrained with MAE for default n_pretrain epochs  and then finetuned. decoder_type - default: seg net - options [SETR, ViT]
    # supervised train a model from scratch for image segmentation


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_mode', choices = ['pretrain','finetune','pruned_pretrain'], default='pretrain')
    parser.add_argument('--prune_method', choices = ['lm','fm'], default="fm")

    parser.add_argument('--config', type=str, default=None, help='use args from a .json config file.')
    
    # masked autoencoder arguments
    ## encoder 
    # todo: get image size from image in loader
    parser.add_argument('--mask_ratio', type=float, default = 0.8,help='proportion of tokens masked')
    # parser.add_argument('--img_size',type = int, default=224, help='img_size H=W resolution of images input to encoder')
    parser.add_argument('--img_size',type = int, default=128, help='img_size H=W resolution of images input to encoder')
    parser.add_argument('--c',type=int, default= 3, help='number of colour channels. default 3 for RGB color')
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--encoder_width', type=int, default =1024,help='embedding dimension for encoder inputs')
    parser.add_argument('--encoder_depth',type=int, default=12)

    
    parser.add_argument('--n_heads', type=int,default=16) # NB embed dim must be divisible by n_heads for multihead attention 
    parser.add_argument('--mlp_ratio', type=int,default=4)
    parser.add_argument('--dropout', type=float, default = 0.1,help='dropout for MLP blocks in transformer blocks')
    
    ## decoder arguments 
    parser.add_argument('--decoder_depth',type =int, default=8)
    parser.add_argument('--decoder_width', type =int, default= 512)
    #for now settings decoder transformer blocks are the same
    
    
    # class tokens for encoder and decoder - open question for whether this helps for when final task is semantic segmentation
    parser.add_argument('--no_cls_token_encoder',action='store_true', help= 'No cls token prepended to embedded token inputsfor the encoder.')
    parser.add_argument('--no_cls_token_decoder',action='store_true', help= 'No cls token prepended to embedded token inputs for the encoder.')

    parser.add_argument('--n_epochs',type=int, default =10)
    parser.add_argument('--batch_size',type=int, default=16)

    parser.add_argument("--imagenet",action='store_true')
    parser.add_argument("--partial_imagenet",action='store_true')
    # add an argument for a differnt test batch size


    args = parser.parse_args()
    return args


def do_mae_training(args):
    checkpoint_dir_base = './models/MAE'
    args.run_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[-7:]
    checkpoint_dir = os.path.join(checkpoint_dir_base, args.run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # create loaders, implement transforms
    trainer = Trainer(args)
    args.mean_pixels = trainer.mean_pixels
    args.std_pixels = trainer.std_pixels
    print(f'Initialized_trainer')
    
    # useful to examine for each run, needed for plot_train
    save_config(args)
    trainer.train_model()


if __name__ == "__main__":
    args = args_parser()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_mode == "pruned_pretrain":
        retain_save_path = os.path.join("dataprune_saves",args.prune_method,"retained")
        pruned_save = os.listdir(retain_save_path)
        for retained_classes in pruned_save:
            # creating checkpoint and run_id based on date dand time 
            args.retained_classes_dir = os.path.join(retain_save_path,retained_classes)
            do_mae_training(args)
    else:
        do_mae_training(args)
