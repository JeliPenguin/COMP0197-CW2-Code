from torchvision.models import resnet18
import torch
from src.loaders.oxfordpets_loader import custom_augmented_oxford_pets
from src.loaders.imagenet_loader import get_hugging_face_loaders
from tqdm import tqdm
import time
import os
import json
import urllib
import argparse
import time

class MDP():
    """
    Mapping based Data Pruning on ImageNet
    """
    def __init__(self,args) -> None:
        
        self.args = args
        print("Using: ",self.args.device)

        self.surrogate_model = resnet18(weights="DEFAULT").to(self.args.device)
        self.surrogate_model.eval()

        # Get the directory of the script being run
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.saves_directory = os.path.join(project_root, 'saves')
        os.makedirs(self.saves_directory, exist_ok=True)

        self.imagenet_classes_num = 1000

        # For testing
        trainset,_ = custom_augmented_oxford_pets(self.args.image_size)
        self.data_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=self.args.batch_size,
                                                shuffle=False)
        
        # Actual loader for imagenet pruning
        # config = {
        #     "img_size":self.args.image_size,
        #     "batch_size":self.args.batch_size,
        #     "imagenet":True
        # }
        # loader_args = argparse.Namespace(**config)
        # self.data_loader, _, _, _ = get_hugging_face_loaders(loader_args)
    
    def prune(self):
        raise NotImplementedError



class LMDP(MDP):
    def __init__(self,args) -> None:
        super().__init__(args)
        # Download ImageNet labels
        imagenet_labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        class_idx = json.loads(urllib.request.urlopen(imagenet_labels_url).read().decode())
        self.idx2label = {int(key): value[1] for key, value in class_idx.items()}
        self.lm_pred_dist_save = os.path.join(self.saves_directory, 'lm_pred_dist.pt')

        print("Performing Label Mapping-based DP")

    def lmdp(self):  
        if self.args.reprune:

            preds = []
            # pbar = tqdm(data_loader, desc='Inferencing', ncols=120, total=len(data_loader))

            for i,(inputs, targets) in enumerate(self.data_loader):

                start_time = time.time()
                inputs  = inputs.to(self.args.device)

                targets = targets.to(self.args.device)

                with torch.no_grad():
                    preds.append(torch.argmax(self.surrogate_model(inputs), dim=-1))
                
                T = time.time() - start_time
                print(f'[Elapsed time:{T:0.1f}][Batch: {i}]')

            preds = torch.cat(preds).cpu()
            pred_dist = torch.Tensor([(preds == label).sum() for label in range(self.imagenet_classes_num)]).int()

            torch.save(pred_dist,self.lm_pred_dist_save)

        self.get_pruned_labels()


    def get_pruned_labels(self):
        pred_dist = torch.load(self.lm_pred_dist_save)
        for retain_class_num in self.args.retain_class_nums:
            print("-"*50)
            print(f"RETAINED {retain_class_num} classes:\n")
            top_classes = torch.topk(pred_dist, k=retain_class_num, largest=True).indices
            for class_idx in top_classes:
                print(self.idx2label[class_idx.item()])
            
            retain_class_dir = os.path.join(self.saves_directory, f'lm_{retain_class_num}_retained.pt')
            torch.save(top_classes,retain_class_dir)
    
    def prune(self):
        self.lmdp()
        self.get_pruned_labels()
        

class FMDP(MDP):
    def __init__(self, args) -> None:
        super().__init__(args)
        print("Performing Feature Mapping-based DP") 
        self.lm_pred_dist_save = os.path.join(self.saves_directory, 'fm_pred_dist.pt')