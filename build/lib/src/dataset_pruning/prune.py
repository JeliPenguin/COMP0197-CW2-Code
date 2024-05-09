from torchvision.models import resnet18
import torch
from src.loaders.oxfordpets_loader import custom_augmented
from src.loaders.imagenet_loader import get_hugging_face_loaders
from tqdm import tqdm
import json
import urllib


class MDP():
    """
    Mapping based Data Pruning
    """
    def __init__(self,args) -> None:
        # Download ImageNet labels
        imagenet_labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        class_idx = json.loads(urllib.request.urlopen(imagenet_labels_url).read().decode())
        self.idx2label = {int(key): value[1] for key, value in class_idx.items()}
        self.args = args
        print("Using: ",self.args.device)

        self.surrogate_model = resnet18(weights="DEFAULT").to(self.args.device)
        self.surrogate_model.eval()

        self.lm_pred_dist_save = "lm_pred_dist.pt"
        self.fm_pred_dist_save = "fm_pred_dist.pt"
    
    def prune(self):
        if self.args.lmdp:
            self.lmdp()
        else:
            self.fmdp()
    
    def fmdp(self):
        print("Performing Feature Mapping-based DP")

    def lmdp(self):
        print("Performing Label Mapping-based DP")
        
        if self.args.reprune:

            trainset,_ = custom_augmented(self.args.image_size)

            data_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=self.args.batch_size,
                                                    shuffle=False)

            fx = []

            pbar = tqdm(data_loader, desc='Inferencing', ncols=120, total=len(data_loader))

            for inputs, targets in pbar:

                inputs  = inputs.to(self.args.device)

                targets = targets.to(self.args.device)

                with torch.no_grad():
                    fx.append(torch.argmax(self.surrogate_model(inputs), dim=-1))

            fx = torch.cat(fx).cpu()
            prediction_distribution = torch.Tensor([(fx == i).sum() for i in range(1000)]).int()

            torch.save(prediction_distribution,self.lm_pred_dist_save)

        self.get_pruned_labels()


    def get_pruned_labels(self):
        prediction_distribution = torch.load(self.lm_pred_dist_save)

        for retain_class_num in self.args.retain_class_nums:
            top_classes = torch.topk(prediction_distribution, k=retain_class_num, largest=True).indices
            # source_train_indices = source_train_labels[tensor_a_in_b(source_train_labels[:, 0], top_classes), 1]
            # source_val_indices = source_val_labels[tensor_a_in_b(source_val_labels[:, 0], top_classes), 1]
            for class_idx in top_classes:
                print(self.idx2label[class_idx.item()])