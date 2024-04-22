from mae_code.mae_arch import MAE
from finetune_decoder import FinetuneDecoder, FinetuneDecoderResnet
import torch
import os
import argparse
import json
from dataloaders import get_finetune_loader,get_hugging_face_loader_OxfordPets,custom_augmented
import time
import os
import time
import torch
import torch.nn as nn
import core as core
import torchmetrics as TM

class Pipeline():
    def __init__(self,args) -> None:
        self.device = torch.device(args.device)
        self.pretrained_encoder = self.load_pretrained_encoder(args.model)
        
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.resnet = args.resnet

        # if args.gen_embed:
        #     self.gen_embedding()

        # self.model = ImageSegmentation(kernel_size=3).to(self.device)
        # self.model = FinetuneDecoder().to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.7)

        # self.train_loader,self.test_loader = get_finetune_loader(self.batch_size,self.img_size)

        self.train()

    def load_config(self,filename='config.json'):
        """
        Loads the configuration from a JSON file.
        """
        with open(filename, 'r') as f:
            config = json.load(f)
            return config
    
    def load_pretrained_encoder(self,model_dir):

        model_config_dir = os.path.join(model_dir,"config.json")
        model_checkpoint_dir = os.path.join(model_dir,"model.pth")

        with open(model_config_dir, 'r') as f:
            model_config = json.load(f)

        args = argparse.Namespace(**model_config)
        
        #unserialize
        self.img_size = model_config['img_size']
        args.mean_pixels = torch.tensor(model_config['mean_pixels'])
        args.std_pixels = torch.tensor(model_config['std_pixels'])

        model = MAE(args)  
        model.load_state_dict(torch.load(model_checkpoint_dir))
        model.to(self.device)

        # common_transform = transforms.Compose([
        #     transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        #     transforms.RandomHorizontalFlip(p=0.5)
        #     # transforms.Normalize(mean=mean, std=std),
        #     # transforms.Normalize(mean=mean, std=std)
        # ])

        # transform = transform_dict
        # transform["common_transform"] = common_transform
        # # transform["post_transform"] = None
        # # transform["post_target_transform"] = None

        return model.encoder_block
    
    def gen_embedding(self):
        # Create dataloader for HF pets dataset:
        if os.listdir("./mae_embeddings/train") and os.listdir("./mae_embeddings/test"):
            print("Embeddings already generated")
            return

        trainloader,testloader,_,_  = get_hugging_face_loader_OxfordPets(self.img_size)
        
        for i, (images, labels,image_ids) in enumerate(trainloader):
            
            print(image_ids)
                    
            images = images.to(self.device)

            #forward pass through the MAE encoder only model
            encoded_features,_,_ = self.pretrained_encoder(images)
            encoded_features = encoded_features.squeeze()
            
            # print("Encoded features shape:", encoded_features.shape)
            
            
            filename = f"./mae_embeddings/train/{image_ids[0]}.pt"  # Saves as a .pt file
            torch.save(encoded_features,filename)
            
        for i, (images, labels,image_ids) in enumerate(testloader):
            
            print(image_ids)
            
                       
            images = images.to(self.device)

            #forward pass through the MAE encoder only model
            encoded_features,_,_ = self.pretrained_encoder(images)
            encoded_features = encoded_features.squeeze()
            
            #print("Encoded features shape:", encoded_features.shape)
            
            filename = f"./mae_embeddings/test/{image_ids[0]}.pt"  # Saves as a .pt file
            torch.save(encoded_features,filename)
    
    
    def train_model(self,epoch,model, loader, optimizer):
        """
        This function trains the model for a single epoch
        :param [int] epoch number (for output)
        :param model: torch.nn model
        :param loader: torch dataloader
        :param optimizer: optimizer instance
        :return:
        """


        start_time = time.time()
        batch_report_rate = 1 # Report every [batch_report_rate] batches

        #core.to_device(model.train())
        model.to(self.device)

        criterion = nn.CrossEntropyLoss(reduction='mean')

        running_loss = 0.0
        running_samples = 0
        #(inputs,targets)
        for batch_idx, sample in enumerate(loader, 0):
            optimizer.zero_grad()
            #inputs = core.to_device(inputs)
            #targets = core.to_device(targets)
            inputs, targets = sample[0].to(self.device), sample[1].to(self.device)

            encoded,_,_ = self.pretrained_encoder(inputs)
            encoded = encoded.unsqueeze(1)

            outputs = model(encoded)

            # The ground truth labels have a channel dimension (NCHW).
            # We need to remove it before passing it into
            # CrossEntropyLoss so that it has shape (NHW) and each element
            # is a value representing the class of the pixel.
            targets = targets.squeeze(dim=1)
            # end if

            # print(outputs.dtype,targets.dtype)
            # print(outputs.shape,targets.shape)
            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_samples += targets.size(0)
            running_loss += loss.item()
        # end for
            if batch_idx % batch_report_rate == 0:
                print("[Epoch: {}][Batch: {}][TimePerSample (s) = {:.3f}][Trained samples: {}][Loss: {:.4f}]".format(
        epoch,
                batch_idx,
                (time.time() - start_time)/running_samples,
                running_samples,
                running_loss / (batch_idx + 1),

        ))
                

    def train(self):
        trainset,testset = custom_augmented(self.img_size)

        train_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

        test_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False)
        
        (test_inputs, test_targets) = next(iter(train_loader))
        test_inputs_encoded,_,_ = self.pretrained_encoder(test_inputs.to(self.device))
        test_inputs_encoded = test_inputs_encoded.unsqueeze(1)

        if self.resnet:
            print("Using Resnet")
            model = FinetuneDecoderResnet(output_channels=3).to(self.device)
        else:
            print("Using conv")
            model = FinetuneDecoder(output_channels=3).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)


        # if not os.path.exists(output_path):
        #     # Create the directory
        #     os.makedirs(output_path)

        start_time = time.time()
        for epoch in range(self.n_epochs):
            T = time.time() - start_time
            print(f"[Elapsed time:{T:0.1f}][Epoch: {epoch:02d}][Learning Rate: {optimizer.param_groups[0]['lr']}]")
            self.train_model(epoch,model, train_loader, optimizer)
            torch.save(model.state_dict(), os.path.join("finetune.pt"))
            with torch.inference_mode():
                # Test set performance report #
                #core.to_device(model.eval())
                model.to(self.device)
                model.eval()
                model_predictions = model(core.to_device(test_inputs_encoded))
                labels = core.to_device(test_targets)
                # print("Predictions Shape: {}".format(predictions.shape))
                predictions = nn.Softmax(dim=1)(model_predictions)

                predicted_labels = predictions.argmax(dim=1)
                # Add a value 1 dimension at dim=1
                predicted_labels = predicted_labels.unsqueeze(1)
                # Create prediction for the mask:
                predicted_mask = predicted_labels.to(torch.float)

                iou = core.to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.TrimapClasses.BACKGROUND))
                iou_accuracy = iou(predicted_mask,labels)

                pixel_metric = core.to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
                pixel_accuracy = pixel_metric(predicted_labels,labels)

                report = f'[Epoch: {epoch:02d}] : Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}]'
                print(report)

            torch.save(model.state_dict(), os.path.join("finetune.pt"))

            if scheduler is not None:
                scheduler.step()

            print("")




    
    


