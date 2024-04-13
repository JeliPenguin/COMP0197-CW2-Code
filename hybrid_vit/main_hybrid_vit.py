import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from hybrid_vit import HybridViT

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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
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
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

class OxfordPets(VisionDataset):
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
        mask = Image.open(mask_path).convert("L")

        # original_mask_array = np.array(mask)
        # print("Unique values in the original mask:", np.unique(original_mask_array))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
            # transformed_mask_array = np.array(mask)
            # print("Unique values in the transformed mask:", np.unique(transformed_mask_array))

        return img, mask

    def __len__(self):
        return len(self.images)

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4811, 0.4492, 0.3957], std=[0.2650, 0.2602, 0.2686])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long) - 1)
    ])

    dataset = OxfordPets(root='./', transform=transform, target_transform=target_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("Transformations set up and datasets are ready.")
    
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
    print("Model saved.")
    
    # Final evaluation
    print("Training completed.")

if __name__ == "__main__":
    main()
