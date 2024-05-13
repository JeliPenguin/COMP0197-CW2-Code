# COMP0197-Applied Deep Learning

## Downloading Model Checkpoints
Please use this link to download all our trained model checkpoints:

https://liveuclac-my.sharepoint.com/:f:/r/personal/ucabs55_ucl_ac_uk/Documents/comp0197-cw2-pt/2235431/new_model?csf=1&web=1&e=LcFcpK


## Install your project in editable mode
```
pip install -e .
```

## Login to HuggingFace
To login to huggingface, use the following command and enter your hugging face token:
```
huggingface-cli login
```

If you don't have a valid token, use the provided login script:
```
python hg_login.py
```

## Pre-training MAE

### Normal Pre-training
```
python .\src\mae_code\run_mae.py
```

### View Results from pre-trained MAE
To visualize the reconstructions from MAE You can use the following command:
```
python .\src\mae_code\plot_trained_mae.py [--model <the path to MAE that you trained, defaults to our trained MAE>]
```

Or through our interactive Jupyter Notebook ```view_mae_examples.ipynb```.


## Finetuning MAE

### View Finetuning Results
To see test results of the finetuned semantic segmentation model, you can use the following command:
```

```
Or through our interactive Jupyter Notebook ```view_mae_examples.ipynb```, which also visualizes the metrics monitored during training.


## Fully Supervised SegNet

### Training
To train our SegNet model, you can use the following command:
```
python src/segnet_bm/train.py [--epochs <number of epochs, default 100>] [--dataset_proportion <proportion 0-1, default 1>]
```
This will train a standard SegNet model and a SegNet with Depthwise Separable Convolution

### Testing



### Interactive Notbook
You may also wish to use our interactive Jupyter Notebook ```view_segnet.ipynb``` to view training metrics and for testing the models



## ImageNet Pruning for efficient pretraining

The current code offers both Label Mapping based and Feature Mapping based Dataset Pruning as proposed in https://arxiv.org/abs/2310.08782. The code is adopted from the paper's original repo https://github.com/OPTML-Group/DP4TL 

### Running Pruning

#### Running Label Mapping Based Dataset Pruning
Using the following command:
```
python .\src\dataset_pruning\run_pruning.py [--lmdp] [--reprune] [--retain_class_nums <Number of classes you wish to retain>]
```

#### Running Feature Mapping Based Dataset Pruning
Using the following command:
```
python .\src\dataset_pruning\run_pruning.py [--fmdp] [--reprune] [--retain_class_nums <Number of classes you wish to retain>]
```

#### Additional Note

If you have already pruned previously, then you can ignore the ```--reprune``` tag. The pruned class labels would be saved within the src/dataset_pruning/save folder.


### Pre-training with Dataset Pruning
```
python .\src\mae_code\run_mae.py --train_mode pruned_pretrain
```
