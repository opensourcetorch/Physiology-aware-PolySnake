# Physiology-aware Structure For Coronary Vessel Segmentation




Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md).

## Testing & Visualization


1. train your own model and prepare your own CCTA datasets, please ensure the input size for network is [15 * 96 * 96] or you can change the size in code.
    ```

## Testing & Visualization 



1. Test & Visualize:
    ```
    python run.py --type visualize  --cfg_file configs/physiology-aware.yaml model physiology_aware test.dataset MedicalTeest
    
    ```




## Training



### Training on Medical dataset

```
python train_net.py --cfg_file configs/physiology-aware.yaml model physiology_aware
```




```
