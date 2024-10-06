import torch.nn as nn
import numpy as np
import torch


""" Feed Forward Neural Network """
class SignLangNN(nn.Module):
    def __init__(self, 
                 n_flayers: int, 
                 in_features: int,
                 hidden_size: int,
                 out_features: int
                 ) -> None:
        super().__init__()
        layers = []

        for _ in range(n_flayers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(hidden_size, out_features))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

""" Convolutional Neural Network """

def compute_size_out_conv2d(input_size: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: int = 0, ceil_mode: bool = False):
    raw_res = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / (stride)) + 1
    if ceil_mode:
        return np.ceil(raw_res)
    else:
        return np.floor(raw_res)

def compute_size_out_maxpool2d(input_size: int, kernel_size: int, stride: int | None = None, dilation: int = 1, padding: int = 0, ceil_mode: bool = False):
    if stride is None:
        stride = kernel_size
    return compute_size_out_conv2d(input_size, kernel_size, stride, dilation, padding, ceil_mode)

class SignLangCNN(nn.Module):
    def __init__(self, 
                 n_flayers: int, 
                 n_clayers: int,
                 in_channels: int,
                 hidden_size: int,
                 in_features: int,
                 out_features: int
                 ) -> None:
        super().__init__()
        layers = []

        out_channels = 4
        in_size = in_features
        for i in range(n_clayers):
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3))
            layers.append(nn.ReLU())
            if (i % 2) == 0:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=3))

            in_channels = out_channels

            # calculate dim's
            s_out = compute_size_out_conv2d(input_size=in_size, kernel_size=3) # size_out 
            if (i % 2) == 0:
                s_out = compute_size_out_maxpool2d(input_size=s_out, kernel_size=3, stride=3)   
            in_size = s_out


        in_features = int((in_size * in_size) * out_channels)

        layers.append(nn.Flatten(start_dim=1)) # Flatten after batch
        for _ in range(n_flayers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(hidden_size, out_features))
        # layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = SignLangCNN(n_flayers=3,
                        n_clayers=1,
                        in_channels=3,
                        hidden_size=16,
                        in_features=128,
                        out_features=12)

    in_tensor = torch.randn(1, 3, 128, 128)
    out = model(in_tensor)

