# data
#import modules
import torch
import os
import datetime
import torch
from mae_utils import get_hugging_face_loaders, load_model, mean, std, get_dataloaders
from model import MAE
import torch
import torchvision.transforms as T
import torchvision
# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()

class Trainer:
    def __init__(self,args):
        self.device = args.device

        if args.train_mode == "pretrain":
            self.mae = MAE(args) 
            self.mae.to(self.device)
            self.args = args
        else:
            self.mae, self.args = load_model(args.model_folder_path)
            self.mae.to(self.device)
            args.img_size = self.args.img_size
            self.args.device = args.device
            self.args.run_id = args.run_id
            self.args.batch_size = args.batch_size

        if args.train_mode == "pretrain":
            self.train_loader, self.val_loader, self.mean_pixels, self.std_pixels  = get_hugging_face_loaders(args)
        else:
            self.train_loader, self.val_loader, self.mean_pixels, self.std_pixels  = get_dataloaders(args)
        # also gets means and stds for unnormalizing
        print(f'Created dataset loaders using dataset in {args.dataset}')
        print(self.args.device)
        self.base_lr = 1.5e-4
        self.lr = self.base_lr * (self.args.batch_size/256)
        self.optimizer =torch.optim.AdamW(self.mae.parameters(), lr=self.lr)
        
        self.checkpoint_dir = args.checkpoint_dir


    def train_one_epoch(self, epoch, model, dataloader, optimizer, device, print_freq):
        model.train()
        total_loss = 0
        iter_loss = 0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(self.device)
            #forward pass through the MAE model
            reconstructed, mask_indices = model(images)

            loss = model.loss(images, reconstructed, mask_indices)
            iter_loss += loss.item()
            total_loss += loss.item()

            # gradients and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % print_freq == 0:
                average_loss = iter_loss / print_freq
                print(f'Batch: {i + 1}, Time: {datetime.datetime.now()}, Average Train Loss: {average_loss}')
                iter_loss = 0
                # val_loss = self.validate(model, self.val_loader, self.device)
                # print(f' Iteration {i + 1}, Train Loss: {loss.item()} | Validation Loss:{val_loss}')
        if epoch % 10 == 0:
            reconstructed = model.reconstruct_image(reconstructed)
            targets_grid = torchvision.utils.make_grid(reconstructed.detach().cpu(), nrow=8)
            t2img(targets_grid.detach().cpu()).show()

        return total_loss / (i+1)
    
    def validate(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                reconstructed, mask_indices = model(images)
                loss = model.loss(images, reconstructed, mask_indices)
                total_loss += loss.item()
                count += 1

        return total_loss / count

    def train_model(self, print_freq=10):
        model = self.mae
        num_epochs = self.args.n_epochs
        
        print("start training")
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(epoch, model, self.train_loader, self.optimizer, self.device, print_freq)
            if epoch == 0:
                print(f'Training MAE: \n device{self.args.device} \n dataset {self.args.dataset}')

            val_loss = self.validate(model, self.val_loader, self.device)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

            #  model checkpoint every epoch 
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model.pth')
            torch.save(model.state_dict(), checkpoint_path)