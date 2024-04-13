import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.nn.functional import interpolate
from PIL import Image
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
import numpy as np
import os

from vit import VisionTransformer

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

class OxfordPets(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Filter out hidden files that start with '._'
        self.images = [img for img in sorted(os.listdir(os.path.join(root, "images"))) if img.endswith(".jpg") and not img.startswith("._")]
        self.masks = [m for m in sorted(os.listdir(os.path.join(root, "annotations", "trimaps"))) if m.endswith(".png") and not m.startswith("._")]

        if len(self.images) != len(self.masks):
            print("Warning: The number of images does not match the number of masks.")
        print(f"Number of images: {len(self.images)}, Number of masks: {len(self.masks)}")
        
        # Print first 5 pairs to check for correct alignment
        for i in range(min(5, len(self.images))):  # Print first 5 pairs
            print(f"Image file: {self.images[i]}, Mask file: {self.masks[i]}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        mask_path = os.path.join(self.root, "annotations", "trimaps", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure mask is loaded as grayscale

        # Convert the mask to a tensor and subtract 1
        mask = torch.tensor(np.array(mask), dtype=torch.long).unsqueeze(0) - 1

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        # Squeeze to remove the channel dimension because cross_entropy does not expect it
        mask = mask.squeeze(0)

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
    # Define the transformations for the images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the transformations for the masks
    target_transform = transforms.Compose([
        transforms.Resize((32, 32), interpolation=Image.NEAREST),
    ])

    # Load the Oxford-IIIT Pet Dataset
    dataset = OxfordPets(root='./', transform=transform, target_transform=target_transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Create the VisionTransformer model
    model = VisionTransformer(
        size_image=32,
        size_patch=4,
        channels_in=3,
        dimension_embed=384,
        layers_number=2,
        heads_number=12,
        dimension_hidden=1536,
        dropout_rate=0.0,
        classes_num=3
    )

    # Define the loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Train the model
    for epoch in range(100):
        print(f"Training epoch: {epoch}")
        model.train()
        train_loss = 0
        iou_train = 0
        for inputs, labels in train_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            iou = iou_score(outputs, labels, num_classes=3)
            iou_train += iou
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        iou_train /= len(train_dataloader)

        print(f"Evaluating epoch: {epoch}")
        model.eval()
        val_loss = 0
        iou_val = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                iou = iou_score(outputs, labels, num_classes=3)
                iou_val += iou
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        iou_val /= len(val_dataloader)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train IoU: {iou_train}, Val Loss: {val_loss}, Val IoU: {iou_val}')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(val_dataloader))
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        out = make_grid(inputs)
        imshow(out, title=[f'pred: {x}, true: {y}' for x, y in zip(preds, labels)])
