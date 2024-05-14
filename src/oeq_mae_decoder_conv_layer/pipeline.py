from src.oeq_mae_decoder_conv_layer.mae_parts import Encoder
from src.oeq_mae_decoder_conv_layer.mae_segment import MAESegmentor
import torch
import os
import argparse
import json
from src.oeq_mae_decoder_conv_layer.dataloaders import custom_augmented
import time
import os
import time
import torch
import torch.nn as nn
import src.oeq_mae_decoder_conv_layer.core as core
import torchmetrics as TM
import torchvision
from src.oeq_mae_decoder_conv_layer.utils import dice_loss
from torch import optim
import matplotlib.pyplot as plt
from src.utils.core import load_model_from_checkpoint

class DefaultArgs:
    def __init__(self):
        self.img_size = 128  # adjusted for Tiny ImageNet
        self.patch_size = 4 # You can experiment with smaller sizes, like 8, if desired
        self.encoder_width = 1024  # Adjusted for a smaller model
        self.n_heads = 16  # Fewer heads given the reduced complexity
        self.encoder_depth = 12  # Fewer layers
        self.decoder_width = 512  # Adjusted decoder width
        self.decoder_depth = 12  # Fewer layers in decoder
        self.mlp_ratio = 4.0
        self.dropout = 0.1
        self.mask_ratio = 0.8
        self.no_cls_token_encoder = False
        self.no_cls_token_decoder = False
        self.model = "./models/pretrained_encoder"
        self.checkpoint_dir = "./models/trained_models"
        self.c = 3  # Number of color channels (RGB)
        self.cuda = True
        self.output_display_period = 10
        self.batch_print_period = 5
        self.batch_size = 4
        self.out_indices = [1, 3, 5, 7, 11]

class Pipeline():
    def __init__(self,args=DefaultArgs()) -> None:
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.args = args
        self.args.device = self.device
        self.batch_size = args.batch_size
        self.model = MAESegmentor(args) 
        if self.model:
            self.model.encoder_block = self.load_encoder(self.args.model)
        self.model.to(self.device)
        trainset, testset = custom_augmented(args.img_size)

        self.train_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False)
        

    def load_encoder(self, model_dir):
        model_config_dir = os.path.join(model_dir,"config.json")
        model_checkpoint_dir = os.path.join(model_dir,"model.pt")

        with open(model_config_dir, 'r') as f:
            model_config = json.load(f)

        args = argparse.Namespace(**model_config)
    
        encoder = Encoder(args)
        # encoder.load_state_dict(torch.load(model_checkpoint_dir))

        load_model_from_checkpoint(encoder,model_checkpoint_dir)

        return encoder
    

    def train_model(self, model:MAESegmentor, epoch, loader, criterion, optimizer):
        """
        This function trains the model for a single epoch
        :param [int] epoch number (for output)
        :param model: torch.nn model
        :param loader: torch dataloader
        :param optimizer: optimizer instance
        :return:
        """

        start_time = time.time()
        running_loss = 0
        running_dice = 0
        running_samples = 0
        model.train()
        for batch_idx, sample in enumerate(loader, 0):
            optimizer.zero_grad()
            inputs, targets = sample[0].to(self.device), sample[1].to(self.device)
            outputs, _ = model(inputs) 
            # outputs = targets

            loss = criterion(outputs, targets.float())
            dice = dice_loss(outputs, targets.float())
            loss += dice

            loss.backward()
            optimizer.step()

            # Calculate other evaluation metrics
 
            running_loss += loss.item()
            running_dice  += dice.item()
            running_samples += targets.size(0)
            if batch_idx % self.args.batch_print_period == 0:
                print("[Epoch: {}][Batch: {}][TimePerSample (s) = {:.3f}][Trained samples: {}][Combined Loss: {:.4f}][Dice Loss: {:.4f}]".format(
                epoch,
                batch_idx,
                (time.time() - start_time)/running_samples,
                running_samples,
                running_loss / (batch_idx + 1),
                running_dice / (batch_idx + 1),
                ))        
        if epoch % self.args.output_display_period == 0:
            mask_pred = (outputs > 0.5).float()
            targets_grid = torchvision.utils.make_grid(mask_pred.detach().cpu(), nrow=8)
            core.t2img(targets_grid.detach().cpu()).show()
            targets_grid = torchvision.utils.make_grid(targets.detach().cpu(), nrow=8)
            core.t2img(targets_grid).show()
        return running_loss / (batch_idx + 1), running_dice / (batch_idx + 1)

    def train(self, n_epoch, freeze_encoder = True, freeze_decoder = False):
        # Freeze Model according to training selection
        for param in self.model.decoder.parameters():
            param.requires_grad = not freeze_decoder 
        for param in self.model.encoder.parameters():
            param.requires_grad = not freeze_encoder 

        train_loader, test_loader = self.train_loader, self.test_loader
        criterion = nn.BCELoss().to(self.device)

        (test_inputs, test_targets) = next(iter(test_loader))

        iou = TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.TrimapClasses.BACKGROUND).to(self.device)
        pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro').to(self.device)
        learning_rate = 1e-3
        optimizer = optim.Adam(self.model.parameters(),
                              lr=learning_rate, weight_decay=1e-8, foreach=True)

        self.history = {
            "iou": [],
            "training_loss": [],
            "validation_loss": [],
            "training_dice": [],
            "validation_dice":[],
            "pixel_accuracy": []
        }
        start_time = time.time()
        for epoch in range(n_epoch):
            T = time.time() - start_time
            print(f"[Elapsed time:{T:0.1f}][Epoch: {epoch:02d}][Learning Rate: {optimizer.param_groups[0]['lr']}]")
            training_loss, training_dice = self.train_model(self.model, epoch, train_loader, criterion, optimizer)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.save_name}.pt"))
            with torch.inference_mode():
                # Test set performance report #
                self.model.eval()

                model_predictions = self.model(test_inputs.to(self.device))
                labels = test_targets.to(self.device)
                # compute the Dice score
                validation_loss = criterion(model_predictions, labels.float())
                validation_dice = dice_loss(model_predictions, labels.float())
                validation_loss += validation_dice

                model_predictions = torch.where(model_predictions < 0.5, torch.tensor(0.0), torch.tensor(1.0))
                iou_accuracy = iou(model_predictions,labels)

                pixel_accuracy = pixel_metric(model_predictions,labels)

                report = f'[Epoch: {epoch:02d}] : Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Loss: {validation_loss}, Dice: {validation_dice}]'
                print(report)
            self.history["iou"].append(iou_accuracy.detach().cpu())
            self.history["training_loss"].append(training_loss)
            self.history["validation_loss"].append(validation_loss.detach().cpu())
            self.history["training_dice"].append(training_dice)
            self.history["validation_dice"].append(validation_dice.detach().cpu())
            self.history["pixel_accuracy"].append(pixel_accuracy.detach().cpu())

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"{self.save_name}.pt"))
            torch.save(self.history, os.path.join(self.save_dir, f"{self.save_name}_history.pt"))

    def plot_history(self):
        epochs = range(1, len(self.history['iou']) + 1)
        plt.figure(figsize=(12, 8))

        # Plot IOU
        plt.plot(epochs, self.history['iou'], 'b', label='IOU')

        # Plot training and validation losses
        plt.plot(epochs, self.history['training_loss'], 'r', label='Training Loss')
        plt.plot(epochs, self.history['validation_loss'], 'g', label='Validation Loss')

        # Plot training and validation dice scores
        plt.plot(epochs, self.history['training_dice'], 'c', label='Training Dice')
        plt.plot(epochs, self.history['validation_dice'], 'm', label='Validation Dice')

        # Plot pixel accuracy
        plt.plot(epochs, self.history['pixel_accuracy'], 'y', label='Pixel Accuracy')

        plt.title('Training Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test(self):
        print("Testing finetuned model")
        self.model.eval()

        iou = TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.DoubleClasses.BACKGROUND).to(self.device)
        pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro').to(self.device)

        iou_accuracies = []
        pixel_accuracies = []
        
        with torch.inference_mode():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader, 0):
                inputs  = inputs.to(self.device)

                targets = targets.to(self.device)
                predictions, _ = self.model(inputs)
                predictions = torch.where(predictions < 0.4, torch.tensor(0.0), torch.tensor(1.0))

                iou_accuracy = iou(predictions, targets)
                pixel_accuracy = pixel_metric(predictions, targets)
                iou_accuracies.append(iou_accuracy.item())
                pixel_accuracies.append(pixel_accuracy.item())
                
                del inputs
                del targets
                del predictions
                
            iou_tensor = torch.FloatTensor(iou_accuracies)
            pixel_tensor = torch.FloatTensor(pixel_accuracies)

            print("Test Dataset Accuracy:")
            print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}")

    def show_data_samples(self):
        (test_inputs, test_targets) = next(iter(self.test_loader))

        # Inspecting input images
        input_grid = torchvision.utils.make_grid(test_inputs, nrow=8)
        core.t2img(input_grid).show()

        # Inspecting the segmentation masks corresponding to the input images
        #
        # When plotting the segmentation mask, we want to convert the tensor
        # into a float tensor with values in the range [0.0 to 1.0]. However, the
        # mask tensor has the values (0, 1, 2), so we divide by 2.0 to normalize.
        targets_grid = torchvision.utils.make_grid(test_targets, nrow=8)
        core.t2img(targets_grid).show()

    def load_model_checkpoint(self, path="finetuned_model.pt"):
        # self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir,path)))
        load_model_from_checkpoint(self.model,os.path.join(self.args.checkpoint_dir,path))
        print("Loaded Model")

    def show_examples(self):
        (test_inputs, test_targets) = next(iter(self.test_loader))

        # Inspecting input images
        input_grid = torchvision.utils.make_grid(test_inputs, nrow=8)
        core.t2img(input_grid).show()


        targets_grid = torchvision.utils.make_grid(test_targets, nrow=8)
        core.t2img(targets_grid).show()

        self.model.eval()
        test_inputs = test_inputs.to(self.device)
        predictions, _ = self.model(test_inputs)
        predictions = torch.where(predictions < 0.3, torch.tensor(0.0), torch.tensor(1.0))

        predicted_mask_grid = torchvision.utils.make_grid(predictions, nrow=8)
        core.t2img(predicted_mask_grid ).show()

    def load_config(self,filename='config.json'):
        """
        Loads the configuration from a JSON file.
        """
        with open(filename, 'r') as f:
            config = json.load(f)
            return config
