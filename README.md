## COMP0197-Applied Deep Learning

# Install your project
```
pip install .
```

# Install your project in editable mode
```
pip install -e .
```


# ImageNet Pruning for efficient pretraining

The current code offers both Label Mapping based and Feature Mapping based Dataset Pruning as proposed in https://arxiv.org/abs/2310.08782. The code is adopted from the paper's original repo https://github.com/OPTML-Group/DP4TL 

## Running Label Mapping Based Dataset Pruning
Using the following command as an example, retaining 500 and 600 classes of the ImageNet dataset:
```
python .\src\dataset_pruning\main.py --lmdp --reprune --retain_class_nums 500,600
```


## Running Feature Mapping Based Dataset Pruning
Using the following command as an example, retaining 500 and 600 classes of the ImageNet dataset:
```
python .\src\dataset_pruning\main.py --fmdp --reprune --retain_class_nums 500,600
```

## Additional Note

If you have already pruned previously, then you can ignore the reprune tag. The pruned class labels would be saved within the src/dataset_pruning/save folder. If you have cuda, then you can also speedup the pruning process by specifying the --cuda tag.