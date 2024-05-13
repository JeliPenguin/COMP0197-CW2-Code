"""
Src : https://github.com/dhruvbird/ml-notebooks/blob/main/pets_segmentation/oxford-iiit-pets-segmentation-using-pytorch-segnet-and-depth-wise-separable-convs.ipynb
Training utility code
"""
import os
import time
import torch
import torch.nn as nn
import src.utils.core as core
import argparse
import src.segnet_bm.segnet as segnet
from src.loaders.oxfordpets_loader import augmented
import torchmetrics as TM

class SegNetTrainer():
    def __init__(self,model,save_name,batch_size=16,dataset_proportion=1) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ",self.device)
        print("Training on train set proportion: ",dataset_proportion)

        self.batch_size = batch_size

        self.model = model
        self.model.to(self.device)

        # Optimizer and Scheduler:
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.7)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.save_path = os.path.join("./models/segnet", save_name)

        # Create training set loader
        self.trainset,self.testset = augmented(dataset_proportion)

        self.training_loss = []
        self.validation_loss = []
        self.iou = []
        self.pixel_accuracy = []

    def train_model(self,epoch, loader):
        """
        This function trains the model for a single epoch
        :param [int] epoch number (for output)
        :param model: torch.nn model
        :param loader: torch dataloader
        :return:
        """

        start_time = time.time()
        batch_report_rate = 50 # Report every [batch_report_rate] batches

        self.model.train()
        # end if

        running_loss = 0.0
        running_samples = 0
        #(inputs,targets)
        for batch_idx, sample in enumerate(loader, 0):
            self.optimizer.zero_grad()
            inputs, targets = sample[0].to(self.device), sample[1].to(self.device)
            outputs = self.model(inputs)

            # The ground truth labels have a channel dimension (NCHW).
            # We need to remove it before passing it into
            # CrossEntropyLoss so that it has shape (NHW) and each element
            # is a value representing the class of the pixel.
            targets = targets.squeeze(dim=1)
            # end if

            loss = self.criterion(outputs, targets)

            loss.backward()

            self.optimizer.step()

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
                
        self.training_loss.append(running_loss / (batch_idx + 1))

    # Define training loop. This will train the model for multiple epochs.
    #
    # epochs: A tuple containing the start epoch (inclusive) and end epoch (exclusive).
    #         The model is trained for [epoch[0] .. epoch[1]) epochs.
    #
    def train_loop(self, loader, test_data, epochs):
        """
        This function manages training over multiple epochs.

        :param model:
        :param loader:
        :param test_data:
        :param epochs: A tuple containing the start epoch (inclusive) and end epoch (exclusive).
        The model is trained for [epoch[0] .. epoch[1]) epochs.
        :param save_path:
        :return:
        """

        if not os.path.exists(self.save_path):
            # Create the directory
            os.makedirs(self.save_path)

        test_inputs, test_targets = test_data
        epoch_i, epoch_j = epochs
        start_time = time.time()
        for epoch in range(epoch_i, epoch_j):
            T = time.time() - start_time
            print(f"[Elapsed time:{T:0.1f}][Epoch: {epoch:02d}][Learning Rate: {self.optimizer.param_groups[0]['lr']}]")
            self.train_model(epoch, loader)
            torch.save(self.model.state_dict(), os.path.join(self.save_path,"segnet.pt"))
            with torch.inference_mode():
                # Test set performance report #
                self.model.to(self.device)
                self.model.eval()
                model_predictions = self.model(test_inputs.to(self.device))
                labels = test_targets.to(self.device)

                squeezed_label = labels.squeeze(dim=1)
                val_loss = self.criterion(model_predictions, squeezed_label)

                # print("Predictions Shape: {}".format(predictions.shape))
                predictions = nn.Softmax(dim=1)(model_predictions)

                predicted_labels = predictions.argmax(dim=1)
                # Add a value 1 dimension at dim=1
                predicted_labels = predicted_labels.unsqueeze(1)
                # Create prediction for the mask:
                predicted_mask = predicted_labels.to(torch.float)

                iou = TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=core.TrimapClasses.BACKGROUND).to(self.device)
                iou_accuracy = iou(predicted_mask,labels)

                pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro').to(self.device)
                pixel_accuracy = pixel_metric(predicted_labels,labels)

                report = f'[Epoch: {epoch:02d}] : Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}]'
                print(report)

                self.validation_loss.append(val_loss.item())
                self.iou.append(iou_accuracy.item())
                self.pixel_accuracy.append(pixel_accuracy.item())

            torch.save(self.model.state_dict(), os.path.join(self.save_path, "segnet.pt"))

            torch.save(self.training_loss, os.path.join(self.save_path, "train_loss.pt"))
            torch.save(self.validation_loss, os.path.join(self.save_path, "val_loss.pt"))
            torch.save(self.iou, os.path.join(self.save_path, "iou.pt"))
            torch.save(self.pixel_accuracy, os.path.join(self.save_path, "pixel_acc.pt"))
            

            if self.scheduler is not None:
                self.scheduler.step()

            print("")

    def do_full_training(self,epochs):
        

        

        train_loader = torch.utils.data.DataLoader(self.trainset,
                                                batch_size=self.batch_size,
                                                shuffle=True)

        test_loader = torch.utils.data.DataLoader(self.testset,
                                                batch_size=self.batch_size,
                                                shuffle=False)
        
        

        (test_inputs, test_targets) = next(iter(test_loader))

        # The code below trains both models consecutively. The following measures are displayed:
        #
        # 1. Training Loss
        # 2. Test accuracy metrics for a single batch (16 images) of test images. The following
        #    metrics are computed:
        #   2.1. Pixel Accuracy
        #   2.2. IoU Accuracy (weighted)
        #   2.3. Custom IoU Accuracy
        
        self.train_loop(train_loader, (test_inputs, test_targets), (1, epochs+1))
    
    def metrics_record(self):
        return (self.training_loss, self.validation_loss, self.iou, self.pixel_accuracy)


def standard_segnet_training(epochs=100,dataset_proportion=1):
    # Run standard segnet model:
    model = segnet.ImageSegmentation(kernel_size=3)
    trainer = SegNetTrainer(model,f"segnet_standard_{dataset_proportion}",dataset_proportion=dataset_proportion)
    trainer.do_full_training(epochs)
    return trainer.metrics_record()


def segnet_dsc_training(epochs=100,dataset_proportion=1):
    # Run segnet + DSC:
    model = segnet.ImageSegmentationDSC(kernel_size=3)
    trainer = SegNetTrainer(model,f"segnet_dsc_{dataset_proportion}",dataset_proportion=dataset_proportion)
    trainer.do_full_training(epochs)
    return trainer.metrics_record()


def args_parser(): 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset_proportion', type=float, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    print(args)
    standard_segnet_training(epochs=args.epochs,dataset_proportion=args.dataset_proportion)
    segnet_dsc_training(epochs=args.epochs,dataset_proportion=args.dataset_proportion)
    