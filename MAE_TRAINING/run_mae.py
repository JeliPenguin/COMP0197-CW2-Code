import torch
from train_mae import Trainer
from mae_utils import save_config, calculate_mean_std
import os
import argparse
import datetime

# three modes:
    #pretrain just pretrains an encoder using masked auto encoder approach 
    # for image segmentation

    # finetune: specify an  --encoder path to add a decoder to an already finetuned encoder at encoder path; if none given, the model's will be pretrained with MAE for default n_pretrain epochs  and then finetuned. decoder_type - default: seg net - options [SETR, ViT]
    # supervised train a model from scratch for image segmentation


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_mode', choices = ['pretrain','finetune'], default='pretrain')
    

    parser.add_argument('--colab', action='store_true', help='only use if running code in colab')
    parser.add_argument('--config', type=str, default=None, help='use args from a .json config file.')
    parser.add_argument('--cuda', action='store_true', help='Force to use CUDA if available')

    parser.add_argument('--model_folder_path', help='path to a model file')

    # data 
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    
    # masked autoencoder arguments
    ## encoder 
    # todo: get image size from image in loader
    parser.add_argument('--mask_ratio', type=float, default = 0.8,help='proportion of tokens masked')
    parser.add_argument('--img_size',type = int, default=128, help='img_size H=W resolution of images input to encoder')
    parser.add_argument('--c',type=int, default= 3, help='number of colour channels. default 3 for RGB color')
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--encoder_width', type=int, default =1024,help='embedding dimension for encoder inputs')
    parser.add_argument('--encoder_depth',type=int, default=12)

    
    parser.add_argument('--n_heads', type=int,default=16) # NB embed dim must be divisible by n_heads for multihead attention 
    parser.add_argument('--mlp_ratio', type=int,default=4)
    parser.add_argument('--dropout', type=float, default = 0.1,help='dropout for MLP blocks in transformer blocks')
    parser.add_argument('--train_split', type=float, default = 0.8,help='Train Split Size')
    
    ## decoder arguments 
    parser.add_argument('--decoder_depth',type =int, default=8)
    parser.add_argument('--decoder_width', type =int, default= 512)
    #for now settings decoder transformer blocks are the same
    
    
    # class tokens for encoder and decoder - open question for whether this helps for when final task is semantic segmentation
    parser.add_argument('--no_cls_token_encoder',action='store_true', help= 'No cls token prepended to embedded token inputsfor the encoder.')
    parser.add_argument('--no_cls_token_decoder',action='store_true', help= 'No cls token prepended to embedded token inputs for the encoder.')

    parser.add_argument('--n_epochs',type=int, default =100)
    parser.add_argument('--batch_size',type=int, default=32)

    parser.add_argument("--imagenet",action='store_true')
    parser.add_argument("--download_imagenet",action='store_true')
    # add an argument for a differnt test batch size


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    checkpoint_dir_base = './MAE'

    # creating checkpoint and run_id based on date dand time 
    args.run_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[-7:]
    checkpoint_dir = os.path.join(checkpoint_dir_base, args.run_id)
    print(checkpoint_dir)
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
