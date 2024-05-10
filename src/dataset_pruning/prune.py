from torchvision.models import resnet18
import torch
from src.loaders.oxfordpets_loader import get_hugging_face_loader_OxfordPets
from src.loaders.imagenet_loader import get_hugging_face_loaders
from tqdm import tqdm
import time
import os
import json
import urllib
import numpy as np
import time
import argparse

# Code adapted from Dataset Pruning for Transfer Learning original
# GitHub repo: https://github.com/OPTML-Group/DP4TL

class MDP():
    """
    Mapping based Data Pruning on ImageNet
    """
    def __init__(self,args,method,pred_dist_save) -> None:
        
        self.args = args
        print("Using device: ",self.args.device)

        self.surrogate_model = resnet18(weights="DEFAULT").to(self.args.device)
        self.surrogate_model.eval()

        # Get the directory of the script being run
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.saves_directory = os.path.join(project_root, 'saves')
        os.makedirs(self.saves_directory, exist_ok=True)

        self.imagenet_classes_num = 1000
        self.pets_classes_num = 37

        self.pets_data_loader,_,_,_ = get_hugging_face_loader_OxfordPets(self.args.image_size,self.args.batch_size)
        
        config = {
            "img_size":self.args.image_size,
            "batch_size":self.args.batch_size,
            "imagenet":True
        }
        loader_args = argparse.Namespace(**config)
        self.imagenet_data_loader, _, _, _ = get_hugging_face_loaders(loader_args)

        # Download ImageNet labels
        imagenet_labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        class_idx = json.loads(urllib.request.urlopen(imagenet_labels_url).read().decode())
        self.idx2label = {int(key): value[1] for key, value in class_idx.items()}
        self.method = method
        self.pred_dist_save = os.path.join(self.saves_directory, pred_dist_save)
    
    def prune(self):
        raise NotImplementedError
    
    def get_pruned_labels(self):
        pred_dist = torch.load(self.pred_dist_save)
        for retain_class_num in self.args.retain_class_nums:
            print("-"*50)
            print(f"RETAINED {retain_class_num} classes:\n")
            top_classes = torch.topk(pred_dist, k=retain_class_num, largest=True).indices
            for class_idx in top_classes:
                print(self.idx2label[class_idx.item()])
            
            retain_class_dir = os.path.join(self.saves_directory, f'{self.method}_{retain_class_num}_retained.pt')
            torch.save(top_classes,retain_class_dir)



class LMDP(MDP):
    def __init__(self,args) -> None:
        super().__init__(args,"lm",'lm_pred_dist.pt')
        print("Performing Label Mapping-based DP")

    def gen_label_pred_distribution(self):  
        if self.args.reprune:

            preds = []

            for inputs, targets in tqdm(self.pets_data_loader):
                inputs  = inputs.to(self.args.device)

                targets = targets.to(self.args.device)

                with torch.no_grad():
                    preds.append(torch.argmax(self.surrogate_model(inputs), dim=-1))

            preds = torch.cat(preds).cpu()
            pred_dist = torch.Tensor([(preds == label).sum() for label in range(self.imagenet_classes_num)]).int()

            torch.save(pred_dist,self.pred_dist_save)
    
    def prune(self):
        self.gen_label_pred_distribution()
        self.get_pruned_labels()
        

class FMDP(MDP):
    def __init__(self, args) -> None:
        super().__init__(args,"fm",'fm_pred_dist.pt')
        self.pets_feature_save_path = os.path.join(self.saves_directory,"oxford_pets_class_feature.pt")
        self.imagenet_feature_save_path = os.path.join(self.saves_directory,"imagenet_class_feature.pt")
        self.feature_extraction_setting = {
            "pets": (self.pets_classes_num,self.pets_data_loader,self.pets_feature_save_path),
            "imagenet":(self.imagenet_classes_num,self.imagenet_data_loader,self.imagenet_feature_save_path),
        }
        print("Performing Feature Mapping-based DP")
    
    def calc_distance(self,x,y):
        return torch.norm(x - y, p=2)

    def generate_features(self,dataset):
        class_num,data_loader,save_path = self.feature_extraction_setting[dataset]
        features = {}

        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        feature_check = 'avgpool'
        for n, m in self.surrogate_model.named_modules():
            if n == feature_check:
                m.register_forward_hook(get_features('feats'))

        class_fx = {}
        for i in range(class_num):
            class_fx[i] = []

        for (input, label) in tqdm(data_loader):
            input = input.to(self.args.device)
            with torch.no_grad():
                _ = self.surrogate_model(input)
            for j in range(input.size(0)):
                # Data-wise file
                class_fx[label[j].item()].append(features['feats'][j, :, :, :].flatten().cpu().numpy())
                
            torch.save(class_fx,save_path)
    
    def generate_target_source_features(self):
        print("Generating Class Features for Oxford IIIT")
        self.generate_features("pets")

        print("Generating Class Features for ImageNet")
        self.generate_features("imagenet")
        
        
    
    def fm_selection(self):
        # self.generate_target_source_features()
        ds_feat = torch.load(self.pets_feature_save_path)
        imagenet_feat = torch.load(self.imagenet_feature_save_path)

        mean_imagenet_feat = {}

        print("Processing ImageNet Features")
        for key in tqdm(imagenet_feat.keys()):
            feat = torch.tensor(np.array(imagenet_feat[key])).to(self.args.device)
            mean_imagenet_feat[key] = feat.mean(dim=0)

        imagenet_class_num = len(mean_imagenet_feat.keys())
        prediction_distribution = torch.zeros(imagenet_class_num, device=self.args.device)

        print("Processing Oxford Pets IIIT Features")
        for ds_key in tqdm(ds_feat):
            data_num_per_class = len(ds_feat[ds_key])
            for i in range(data_num_per_class):
                data = torch.tensor(np.array(ds_feat[ds_key][i]), device=self.args.device)
                distance_score = torch.zeros(imagenet_class_num, device=self.args.device).requires_grad_(False)
                for key in imagenet_feat.keys():
                    distance_score[i] = self.calc_distance(data, mean_imagenet_feat[key])
                class_index = distance_score.argmin()
                prediction_distribution[class_index] += 1

        prediction_distribution = prediction_distribution.int()
        torch.save(prediction_distribution,self.pred_dist_save)


    def prune(self):
        if self.args.reprune:
            self.fm_selection() 
        self.get_pruned_labels()