# COMP0197-Applied Deep Learning

## Install your project in editable mode
```
pip install -e .
```
<!-- 
## Build your project
```
pip install .
``` -->


## ImageNet Pruning for efficient pretraining

The current code offers both Label Mapping based and Feature Mapping based Dataset Pruning as proposed in https://arxiv.org/abs/2310.08782. The code is adopted from the paper's original repo https://github.com/OPTML-Group/DP4TL 

### Running Label Mapping Based Dataset Pruning
Using the following command:
```
python .\src\dataset_pruning\run_pruning.py --lmdp --reprune
```


### Running Feature Mapping Based Dataset Pruning
Using the following command:
```
python .\src\dataset_pruning\run_pruning.py --fmdp --reprune
```

### Additional Note

If you have already pruned previously, then you can ignore the ```--reprune``` tag. The pruned class labels would be saved within the src/dataset_pruning/save folder. If you have cuda, then you can also speedup the pruning process by specifying the ```--cuda tag```.

#### Changing the number of classes to be retained

Using the following command as an example, using FMDP, retaining 500 and 600 classes of the ImageNet dataset:
```
python .\src\dataset_pruning\run_pruning.py --fmdp --reprune --retain_class_nums 500,600
```

## Pre-training MAE

### Normal Pre-training
```
python .\src\mae_code\run_mae.py
```

### Pre-training with Dataset Pruning
```
python .\src\mae_code\run_mae.py --train_mode pruned_pretrain
```