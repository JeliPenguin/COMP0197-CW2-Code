import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.nn.functional import interpolate
from PIL import Image
from torchvision.datasets import VisionDataset
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

        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "annotations", "trimaps"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        mask_path = os.path.join(self.root, "annotations", "trimaps", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # Subtract 1 from the mask to match the labels expected by our model
        mask = Image.open(mask_path) - 1

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

    def __len__(self):
        return len(self.images)

def main():
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Load the Oxford-IIIT Pet Dataset
    dataset = OxfordPets(root='./', transform=transform)

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
    for epoch in range(100):  # Number of epochs
        model.train()
        for inputs, labels in train_dataloader:
            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Zero the gradients
            optimizer.zero_grad()

        # Evaluate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss}')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

if __name__ == '__main__':
    main()
