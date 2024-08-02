## 2024-07-24 19:33:25

----------------------------------------------------------------------
- Total Samples: 9680 
- Samples used for Training: 7744 
- Samples used for Testing: 1936 
- Train-Size (\%): 0.8
----------------------------------------------------------------------
- 1506/1936 were predicted correctly
- Resulting in an accuracy of 0.78
----------------------------------------------------------------------
Hyperparameters:
- Batch-Size: 32
- Epochs: 10
- Learning-Rate: 0.001
----------------------------------------------------------------------
Model:
```
 ConvSignLangNN(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 8, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=6728, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=36, bias=True)
  (softmax): Softmax(dim=1)
) 
```
## 2024-07-24 19:54:22

----------------------------------------------------------------------
- Total Samples: 9680 
- Samples used for Training: 7744 
- Samples used for Testing: 1936 
- Train-Size (\%): 0.8
----------------------------------------------------------------------
- 1511/1936 were predicted correctly
- Resulting in an accuracy of 0.78
----------------------------------------------------------------------
Hyperparameters:
- Batch-Size: 32
- Epochs: 30
- Learning-Rate: 0.001
----------------------------------------------------------------------
Model:
```
 ConvSignLangNN(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 8, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=6728, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=36, bias=True)
  (softmax): Softmax(dim=1)
) 
```
##  2024-07-31 12:19:54

----------------------------------------------------------------------
- Total Samples: 9680 
- Samples used for Training: 7744 
- Samples used for Testing: 1936 
- Train-Size (\%): 0.8
----------------------------------------------------------------------
- 1102/1936 were predicted correctly
- Resulting in an accuracy of 56.92%
----------------------------------------------------------------------
Hyperparameters:
- Batch-Size: 32
- Epochs: 10
- Learning-Rate: 0.001
----------------------------------------------------------------------
Model:
```
 ConvSignLangNN(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 8, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=6728, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=120, bias=True)
  (fc3): Linear(in_features=120, out_features=84, bias=True)
  (fc4): Linear(in_features=84, out_features=36, bias=True)
  (softmax): Softmax(dim=1)
) 
```
----
> Using DataAugmentation

