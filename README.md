# Sign Language Image Recognition

## Best Performing Model
TrainSignLang:
- Epochs: 20
- Learning Rate: 0.001
- % Train-Set: 0.8
- Batch Size: 32
--------------------------------------------------------------------------------
TrainSignLang:
- Epochs: 20
- Learning Rate: 0.001
- % Train-Set: 0.8
- Batch Size: 32
--------------------------------------------------------------------------------
model: 
```
ConvSignLangNN_4_(
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 18, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(18, 24, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=3456, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=128, bias=True)
  (fc4): Linear(in_features=128, out_features=36, bias=True)
)
```
--------------------------------------------------------------------------------
Optimizer:
 ```
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
```
--------------------------------------------------------------------------------
- Size Train-Set: 7744
- Size Test-Set: 1936
- Total Size: 9680
- Data Augmentation: False
--------------------------------------------------------------------------------
- Accuracy: 0.914
- Loss: 0.291
--------------------------------------------------------------------------------
