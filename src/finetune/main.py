import argparse
import torch
from src.finetune.pipeline import Pipeline


def args_parser():
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--encoder_width', type=int, default=1024, help='Encoder width')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--encoder_depth', type=int, default=12, help='Encoder depth')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mask_ratio', type=float, default=0.8, help='Mask ratio')
    parser.add_argument('--no_cls_token_encoder', action='store_true', help='Exclude class token in encoder')
    parser.add_argument('--no_cls_token_decoder', action='store_true', help='Exclude class token in decoder')
    parser.add_argument('--c', type=int, default=3, help='Number of color channels')
    parser.add_argument('--checkpoint_dir', type=str, default="./models/trained_models", help='Model store path')
    parser.add_argument('--model', type=str, default="./models/pretrained_encoder", help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--report_rate', type=int, default=2, help='Report rate')
    parser.add_argument('--skip_conn', action='store_true', help='Use skip connections')
    parser.add_argument('--output_display_period', type=int, default=10, help='Output display period')
    parser.add_argument('--batch_print_period', type=int, default=5, help='Batch print period')
    parser.add_argument('--n_epoch', type=int, default=5, help='Batch print period')
    parser.add_argument('--finetune_percentage', type=float, default=1.0,help='Percentage of finetune dataset to train on')
    parser.add_argument("--refinetune",action='store_true', help='Re-finetune model')
    parser.add_argument("--save_name",type=str, default="finetune.pt")
    return parser.parse_args()


def do_fine_tune(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = Pipeline(args)

    print(f'Initialized Pipeline')

    if args.refinetune:
        print(f"FINE TUNE DECODER {args.n_epoch} EPOCH")
        pipeline.train(n_epoch=args.n_epoch, freeze_encoder=False, freeze_decoder=False)
    else:
        pipeline.load_model_checkpoint("./models/trained_models/100_epoch.pt")
    
    pipeline.test()
    pipeline.show_examples()

    
if __name__ == "__main__":
    args = args_parser()
    
    do_fine_tune(args)
