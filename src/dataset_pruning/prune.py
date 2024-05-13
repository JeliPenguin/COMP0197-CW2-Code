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
    Class for implementing Mapping based Data Pruning (MDP) on ImageNet dataset.
    This class sets up the environment and models for pruning ImageNet data based on specified criteria.
    """

    def __init__(self, args, method) -> None:
        # Initialize the MDP instance with configurations and the pruning method.
        
        # Storing the command line arguments in the instance for easy access
        self.args = args
        print("Using device: ", self.args.device)

        # Setup a surrogate model (ResNet-18) and move it to the appropriate device, set to evaluation mode
        self.surrogate_model = resnet18(weights="DEFAULT").to(self.args.device)
        self.surrogate_model.eval()

        # Number of classes in ImageNet and Oxford-IIIT Pets dataset
        self.imagenet_classes_num = 1000
        self.pets_classes_num = 37

        # Data loader for the Pets dataset
        self.pets_data_loader, _, _, _ = get_hugging_face_loader_OxfordPets(self.args.image_size, self.args.batch_size)
        
        # Setup data loader configuration for ImageNet
        config = {
            "img_size": self.args.image_size,
            "batch_size": self.args.batch_size,
            "imagenet": True
        }
        loader_args = argparse.Namespace(**config)
        self.imagenet_data_loader, _, _, _ = get_hugging_face_loaders(loader_args)

        # Download ImageNet class labels and convert to dictionary mapping IDs to labels
        imagenet_labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        class_idx = json.loads(urllib.request.urlopen(imagenet_labels_url).read().decode())
        self.idx2label = {int(key): value[1] for key, value in class_idx.items()}

        # Storing the method specified for pruning
        self.method = method

        # Setting up directories for saving data during pruning
        project_root = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script being run
        saves_directory = 'dataprune_saves'  # Directory for saving output files
        os.makedirs(saves_directory, exist_ok=True)
        self.dp_save_dir = os.path.join(saves_directory, method)
        os.makedirs(self.dp_save_dir, exist_ok=True)
        self.pred_dist_save = os.path.join(self.dp_save_dir, 'pred_dist.pt')
        self.retained_save = os.path.join(self.dp_save_dir, "retained")
        os.makedirs(self.retained_save, exist_ok=True)
    
    def prune(self):
        # This function should implement the actual pruning logic (currently not implemented)
        raise NotImplementedError
    
    def get_pruned_labels(self):
        # Load predicted distributions from saved file
        pred_dist = torch.load(self.pred_dist_save)
        
        # Process each number of classes to retain specified in command line arguments
        for retain_class_num in self.args.retain_class_nums:
            print("-" * 50)
            print(f"RETAINED {retain_class_num} classes:\n")
            # Identify top classes based on the distribution
            top_classes = torch.topk(pred_dist, k=retain_class_num, largest=True).indices
            print(top_classes)
            # Print labels of the top classes
            for class_idx in top_classes:
                print(self.idx2label[class_idx.item()])
            
            # Save the indices of retained classes to a file
            retain_class_dir = os.path.join(self.retained_save, f'{retain_class_num}_classes.pt')
            torch.save(top_classes, retain_class_dir)




class LMDP(MDP):
    """
    Subclass of MDP that specializes in Label Mapping-based Data Pruning (LMDP).
    This class utilizes label prediction distributions from a surrogate model to determine pruning decisions.
    """

    def __init__(self, args) -> None:
        # Initialize the superclass with specific arguments and the pruning method 'lm' for label mapping
        super().__init__(args, "lm")
        print("Performing Label Mapping-based DP")

    def gen_label_pred_distribution(self):
        """
        Generates a label prediction distribution for ImageNet classes based on outputs from a surrogate model.
        This method is crucial for identifying how frequently each class is predicted, forming the basis for pruning.
        """
        if self.args.reprune:
            # Only generate the distribution if re-pruning is specified in the arguments

            preds = []  # List to collect predictions

            # Loop over the dataset of pet images
            for inputs, targets in tqdm(self.pets_data_loader):
                # Move inputs and targets to the specified device (e.g., GPU)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                # Perform inference without computing gradients
                with torch.no_grad():
                    # Append the predicted classes (as indices) to the list
                    preds.append(torch.argmax(self.surrogate_model(inputs), dim=-1))

            # Concatenate all predictions into a single tensor and move it to CPU memory
            preds = torch.cat(preds).cpu()
            # Count occurrences of each ImageNet class label in the predictions
            pred_dist = torch.Tensor([(preds == label).sum() for label in range(self.imagenet_classes_num)]).int()

            # Save the computed distribution to file
            torch.save(pred_dist, self.pred_dist_save)
    
    def prune(self):
        """
        Executes the pruning process by first generating a label prediction distribution,
        then determining which labels to retain based on the distribution.
        """
        self.gen_label_pred_distribution()  # Generate the distribution
        self.get_pruned_labels()            # Retrieve and save the labels that meet the pruning criteria

        

class FMDP(MDP):
    """
    Subclass of MDP that specializes in Feature Mapping-based Data Pruning (FMDP).
    This class utilizes features extracted from a surrogate model to determine pruning decisions based on feature similarity.
    """

    def __init__(self, args) -> None:
        # Initialize the superclass with specific arguments and the pruning method 'fm' for feature mapping
        super().__init__(args, "fm")
        # Define paths to save features of different datasets
        self.pets_feature_save_path = os.path.join(self.dp_save_dir, "oxford_pets_class_feature.pt")
        self.imagenet_feature_save_path = os.path.join(self.dp_save_dir, "imagenet_class_feature.pt")
        # Setup configurations for feature extraction of both datasets
        self.feature_extraction_setting = {
            "pets": (self.pets_classes_num, self.pets_data_loader, self.pets_feature_save_path),
            "imagenet": (self.imagenet_classes_num, self.imagenet_data_loader, self.imagenet_feature_save_path),
        }
        print("Performing Feature Mapping-based DP")
    
    def calc_distance(self, x, y):
        # Calculate the Euclidean distance between two feature vectors
        return torch.norm(x - y, p=2)

    def generate_features(self, dataset):
        # Extract features for a specified dataset and save them
        class_num, data_loader, save_path = self.feature_extraction_setting[dataset]
        features = {}
        
        def get_features(name):
            # Define a forward hook to capture output features from a specified layer
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        # Name of the layer from which to capture features
        feature_check = 'avgpool'
        for n, m in self.surrogate_model.named_modules():
            if n == feature_check:
                m.register_forward_hook(get_features('feats'))

        class_fx = {}
        for i in range(class_num):
            class_fx[i] = []

        # Iterate over the dataset, capturing and storing features
        for (input, label) in tqdm(data_loader):
            input = input.to(self.args.device)
            with torch.no_grad():
                _ = self.surrogate_model(input)
            for j in range(input.size(0)):
                class_fx[label[j].item()].append(features['feats'][j, :, :, :].flatten().cpu().numpy())
                
            torch.save(class_fx, save_path)
    
    def generate_target_source_features(self):
        # Generate and save features for both datasets
        print("Generating Class Features for Oxford IIIT")
        self.generate_features("pets")
        print("Generating Class Features for ImageNet")
        self.generate_features("imagenet")
        
    def fm_selection(self):
        # Perform feature mapping based selection to determine which classes to prune
        self.generate_target_source_features()
        ds_feat = torch.load(self.pets_feature_save_path)
        imagenet_feat = torch.load(self.imagenet_feature_save_path)

        mean_imagenet_feat = {}
        # Compute mean features for ImageNet classes
        print("Processing ImageNet Features")
        for key in tqdm(imagenet_feat.keys()):
            feat = torch.tensor(np.array(imagenet_feat[key])).to(self.args.device)
            mean_imagenet_feat[key] = feat.mean(dim=0)

        imagenet_class_num = len(mean_imagenet_feat.keys())
        prediction_distribution = torch.zeros(imagenet_class_num, device=self.args.device)

        # Compare features from Oxford IIIT Pets dataset to ImageNet mean features and update prediction distribution
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
        torch.save(prediction_distribution, self.pred_dist_save)

    def prune(self):
        # Execute pruning if re-pruning is requested
        if self.args.reprune:
            self.fm_selection()
        self.get_pruned_labels()
