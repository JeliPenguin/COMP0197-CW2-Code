from src.segnet_bm.train import segnet_dsc_training,standard_segnet_training
from src.finetune.main import do_fine_tune,args_parser
import torch

dataset_proportions = [0.05,0.1,0.5,0.8,1]
args = args_parser()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for p in dataset_proportions:
    segnet_dsc_training(epochs=50,dataset_proportion=p)
    standard_segnet_training(epochs=50,dataset_proportion=p)
    
    args.finetune_percentage = p
    args.save_name = f"mae_finetune_{p}"
    args.refinetune = True
    args.n_epoch = 50
    
    do_fine_tune(args)