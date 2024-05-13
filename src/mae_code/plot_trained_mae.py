import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.mae_code.mae_utils import load_model
from src.loaders.imagenet_loader import get_hugging_face_loaders,mean,std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_comparisons(args,model):
    #compares masked original, image, autoencoder reconstruction
    # on a batch of images
    
    _, test_loader, _, _ = get_hugging_face_loaders(args)
   
    model.to(device)
    model.eval()
    with torch.no_grad():
          batch = next(iter(test_loader))
          images = batch[0].to(device)


          decoder_output, mask_idxs = model(images)
          reconstructions = model.reconstruct_image(decoder_output)
        
          masks = model.create_visual_mask(images, mask_idxs, args.patch_size)

          masked_images = images * masks

          #the number of examples to display (limited to a manageable number for visualization)
          batch_size = images.size(0)
          if batch_size > 4:
              print("Batch size is too large for effective visualization. Reducing to 4 for display.")
              batch_size = 4  # Adjust batch size here if needed

          fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))  # 3 columns for each type of image

          # no. of rows in plot is the no. of samples in batch
          for i in range(batch_size):
              imshow(masked_images[i], axes[i, 0])
              imshow(reconstructions[i], axes[i, 1])
              imshow(images[i], axes[i, 2])

          # Labeling columns
          columns = ['Masked Image', 'Reconstruction', 'Original Image']
          for ax, col in zip(axes[0], columns):
              ax.set_title(col)

          plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between images
          plt.show()
           # Show only one batch for demonstration

def imshow(img, ax):
    # Helper function to unnormalize and show an image on a given Axes object.
    mean_t = mean.view(3, 1, 1).to(device)
    std_t = std.view(3, 1, 1).to(device)
    img = img * std_t + mean_t  # Unnormalize and move to CPU
    npimg = img.cpu().numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert from Tensor image
    ax.axis('off')  # Hide axes ticks

def view_mae_results(model_path):
    model, old_args = load_model(model_path)

    old_args.imagenet = True
    print(old_args)

    visualize_comparisons(old_args,model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="./models/MAE/00000", help='path to a model file')

    args = parser.parse_args()

    view_mae_results(args.model)