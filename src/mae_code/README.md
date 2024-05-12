# self supervised masked autoencoder pretraining of vision models for semantic segmentation

This PyTorch repo provides tools and scripts for training and testing self-supervised masked autoencoder(MAE) pretraining of vision transformer encoders,


## brief file guide


- **`run_mae.py`**: Script to train the MAE. Supports various configurations and handles both CPU and GPU computations. 
    - It also handles running in colab quite automatically. simply use the colab argument when running run.py. 
    - Make sure to  use `--cuda` if you want to train the model on a gpu.

- **`train_mae.py`**: Contains the `Trainer` class which manages the training process, data loading, and checkpoint saving.
- **`plot_trained_mae.py`**: Visualizes outputs of the trained model by comparing original images, incomplete input to the MAE and its reconstruction. 
- **`mae_utils.py`**: Includes utilities for data loading, configuration saving, and calculating dataset statistics for normalization and reversing normalization.

Ensure the following dependencies are installed:

- Python 3.6+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy

Install these with the following command:
```bash
pip install torch torchvision matplotlib numpy
```

### Install Git LFS:
```bash
sudo apt-get install git-lfs
git lfs install
```

## usage
### training
`python run_mae.py --dataset /path/to/your/dataset'

- `--train_mode` (default: `pretrain`): Currently supports masked autoencoding.
- `--dataset`: Path to the dataset.
- `--batch_size` (default: `256`): Batch size for training.
- `--img_size` (default: `224`): Dimensions of images to process. Ensure that this makes sense for the input dataset. 


### visualizing model reconstruction
To visualize your MAEs reconstruction compared with the masked image and ground truth:
`python plot_trained_mae.py --model path/to/your/model.pth --config path/to/your/config.json --dataset /path/to/your/dataset`










