import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import resize
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

from hybrid_vit import HybridViT

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

class OxfordPets(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.images = [img for img in sorted(os.listdir(os.path.join(root, "images")))
                       if img.endswith(".jpg") and not img.startswith("._")]
        self.masks = [mask for mask in sorted(os.listdir(os.path.join(root, "annotations", "trimaps")))
                      if mask.endswith(".png") and not mask.startswith("._")]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        mask_path = os.path.join(self.root, "annotations", "trimaps", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.tensor(np.array(mask), dtype=torch.long) - 1  # Convert mask to tensor and adjust labels directly here

        return img, mask

    def __len__(self):
        return len(self.images)

def iou_score(output, target, num_classes=3):
    smooth = 1e-6
    ious = []
    output = torch.argmax(output, dim=1).data.cpu().numpy()
    target = target.data.cpu().numpy()

    for cls in range(num_classes):
        output_ = output == cls
        target_ = target == cls
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return sum(ious) / num_classes

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def main():
    # Transformations for the input images and masks
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        lambda x: x.squeeze(0),  # Remove channel dimension if added by ToTensor
        lambda x: x.to(dtype=torch.long)  # Convert to long type without warning
    ])

    # Initialize dataset
    dataset = OxfordPets(root='./', transform=transform, target_transform=target_transform)
    
    # Splitting dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoader setup
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Model setup
    model = HybridViT(
        image_size=256,
        patch_size=16,
        stride=16,  # Matching stride to patch size for non-overlapping patches
        num_classes=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        feedforward_dim=3072
    )
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Training loop
    for epoch in range(100):
        print(f"Training epoch: {epoch}")
        model.train()
        train_loss, train_iou = 0, 0
        
        for inputs, masks in train_dataloader:
            inputs, masks = inputs.to(device), masks.to(device).long()  # Ensure masks are Long type
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += iou_score(outputs, masks, num_classes=3)
        
        train_loss /= len(train_dataloader)
        train_iou /= len(train_dataloader)
        
        # Validation loop
        print(f"Evaluating epoch: {epoch}")
        model.eval()
        val_loss, val_iou = 0, 0
        with torch.no_grad():
            for inputs, masks in val_dataloader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_iou += iou_score(outputs, masks, num_classes=3)
        
        val_loss /= len(val_dataloader)
        val_iou /= len(val_dataloader)
        
        # Print stats
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train IoU = {train_iou:.4f}, Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}")
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    # Final evaluation (could include more detailed analysis, visualization, etc.)
    print("Training completed. Model saved and ready for further validation or deployment.")
