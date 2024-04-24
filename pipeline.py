from mae_code.mae_arch import MAE
from finetune_decoder import FinetuneDecoder, FinetuneDecoderResnet
import torch
import os
import argparse
import json
from dataloaders import get_hugging_face_loader_OxfordPets,custom_augmented
import time
import os
import time
import torch
import torch.nn as nn
import core as core
import torchmetrics as TM
import torchvision
from torchvision import models

class Pipeline():
    def __init__(self,args) -> None:
        self.device = torch.device(args.device)
        self.resnet = args.resnet
        self.facebook_mae = args.facebook_mae
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.pretrained_encoder = self.load_pretrained_encoder(args.model)

        
        if self.resnet:
            print("Using Resnet")
            self.model = FinetuneDecoder(input_channels=2048,output_channels=3,output_size=self.img_size).to(self.device)
        # elif self.facebook_mae:
        #     print("Using Facebook MAE")
        #     self.model = FinetuneDecoder(input_channels=2048,output_channels=3).to(self.device)
        else:
            self.model = FinetuneDecoder(output_channels=3,output_size=self.img_size).to(self.device)
            # self.model = FeedForwardDecoder(12,1024).to(self.device)

        # if args.gen_embed:
        #     self.gen_embedding()

        # self.model = ImageSegmentation(kernel_size=3).to(self.device)
        # self.model = FinetuneDecoder().to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.7)

        # self.train_loader,self.test_loader = get_finetune_loader(self.batch_size,self.img_size)

        trainset,testset = custom_augmented(self.img_size)

        self.train_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False)
        
    def load_model(self,model_dir):
        model_config_dir = os.path.join(model_dir,"config.json")
        model_checkpoint_dir = os.path.join(model_dir,"model.pth")

        with open(model_config_dir, 'r') as f:
            model_config = json.load(f)

        args = argparse.Namespace(**model_config)
        
        self.img_size = model_config['img_size']

        if self.resnet:
            model = models.resnet50(pretrained=True).to(self.device)
        # elif self.facebook_mae:
        #     chkpt_dir = 'MAE/mae_visualize_vit_large.pth'
        #     model = getattr(chkpt_dir, 'mae_vit_large_patch16')()
        #     checkpoint = torch.load(chkpt_dir, map_location='cpu')
        #     model.load_state_dict(checkpoint['model'], strict=False).to(self.device())
        else:     
            args.mean_pixels = torch.tensor(model_config['mean_pixels'])
            args.std_pixels = torch.tensor(model_config['std_pixels'])

            model = MAE(args)  
            model.load_state_dict(torch.load(model_checkpoint_dir))
            
        model.to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def load_pretrained_encoder(self,model_dir):

        model = self.load_model(model_dir)

        if self.resnet:
            encoder = nn.Sequential(*list(model.children())[:-2])
        # elif self.facebook_mae:
        #     print(model)
        #     exit()
        else:
            encoder = model.encoder_block

        return encoder

        
    
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
    
    def predict(self,inputs):
        if self.resnet or self.facebook_mae:
            encoded = self.pretrained_encoder(inputs.to(self.device))
        else:
            encoded,_,_ = self.pretrained_encoder(inputs.to(self.device))
            encoded = encoded.unsqueeze(1)

        return self.model(encoded)

    def train_model(self,epoch, loader, optimizer):
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

        criterion = nn.CrossEntropyLoss(reduction='mean')

        running_loss = 0.0
        running_samples = 0
        #(inputs,targets)
        for batch_idx, sample in enumerate(loader, 0):
            optimizer.zero_grad()
            #inputs = core.to_device(inputs)
            #targets = core.to_device(targets)
            inputs, targets = sample[0].to(self.device), sample[1].to(self.device)

            outputs = self.predict(inputs)

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
        print("Fine tuning")
        (test_inputs, test_targets) = next(iter(self.train_loader))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)


        # if not os.path.exists(output_path):
        #     # Create the directory
        #     os.makedirs(output_path)

        start_time = time.time()
        for epoch in range(self.n_epochs):
            T = time.time() - start_time
            print(f"[Elapsed time:{T:0.1f}][Epoch: {epoch:02d}][Learning Rate: {optimizer.param_groups[0]['lr']}]")
            self.train_model(epoch, self.train_loader, optimizer)
            torch.save(self.model.state_dict(), os.path.join("finetune.pt"))
            with torch.inference_mode():
                # Test set performance report #
                #core.to_device(model.eval())
                
                self.model.eval()
                model_predictions = self.predict(test_inputs)
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

            torch.save(self.model.state_dict(), os.path.join("finetune.pt"))

            if scheduler is not None:
                scheduler.step()

            print("")

    def test(self):
        print("Testing finetuned model")
        self.model.load_state_dict(torch.load("finetune.pt"))
        self.model.eval()

        iou = core.to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.TrimapClasses.BACKGROUND))
        pixel_metric = core.to_device(TM.classification.MulticlassAccuracy(3, average='micro'))

        iou_accuracies = []
        pixel_accuracies = []

        with torch.inference_mode():

            for batch_idx, (inputs, targets) in enumerate(self.test_loader, 0):

                inputs  = core.to_device(inputs)

                targets = core.to_device(targets)
                predictions = self.predict(inputs)

                pred_probabilities = nn.Softmax(dim=1)(predictions)
                pred_labels = predictions.argmax(dim=1)

                # Add a value 1 dimension at dim=1
                pred_labels = pred_labels.unsqueeze(1)
                # print("pred_labels.shape: {}".format(pred_labels.shape))
                pred_mask = pred_labels.to(torch.float)

                iou_accuracy = iou(pred_mask, targets)
                # pixel_accuracy = pixel_metric(pred_mask, targets)
                pixel_accuracy = pixel_metric(pred_labels, targets)
                iou_accuracies.append(iou_accuracy.item())
                pixel_accuracies.append(pixel_accuracy.item())

                del inputs
                del targets
                del predictions
            # end for

            iou_tensor = torch.FloatTensor(iou_accuracies)
            pixel_tensor = torch.FloatTensor(pixel_accuracies)

            print("Test Dataset Accuracy:")
            print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}")
        
    
    def show_examples(self):
        (test_inputs, test_targets) = next(iter(self.test_loader))

        # Inspecting input images
        input_grid = torchvision.utils.make_grid(test_inputs, nrow=8)
        core.t2img(input_grid).show()

        # Inspecting the segmentation masks corresponding to the input images
        #
        # When plotting the segmentation mask, we want to convert the tensor
        # into a float tensor with values in the range [0.0 to 1.0]. However, the
        # mask tensor has the values (0, 1, 2), so we divide by 2.0 to normalize.
        targets_grid = torchvision.utils.make_grid(test_targets / 2.0, nrow=8)
        core.t2img(targets_grid).show()


        # Get segmentation mask predicted by the model:
        self.model.load_state_dict(torch.load("finetune.pt"))
        self.model.eval()

        predictions = self.predict(test_inputs)
        # Apply softmax
        predictions = nn.Softmax(dim=1)(predictions)
        # Get label - max probability (ties get broken as first element with max value)
        predicted_labels = predictions.argmax(dim=1)
        # Add a value 1 dimension at dim=1
        predicted_labels = predicted_labels.unsqueeze(1)
        # print("pred_labels.shape: {}".format(pred_labels.shape))
        predicted_mask = predicted_labels.to(torch.float)

        predicted_mask_grid = torchvision.utils.make_grid(predicted_mask / 2.0, nrow=8)
        core.t2img(predicted_mask_grid ).show()




    
    


