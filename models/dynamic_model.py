import torch.nn as nn
import optuna
import numpy as np
import torch
import time


""" Feed Forward Neural Network """
class SignLangNN(nn.Module):
    def __init__(self, 
                 n_flayers: int, 
                 in_features: int, hidden_size: int,
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
        return int(np.ceil(raw_res))
    else:
        return int(np.floor(raw_res))

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
                 out_features: int,
                 trial: optuna.Trial, 
                 ) -> None:
        super().__init__()
        layers = []

        in_size = in_features
        for i in range(n_clayers):
            # suggest params
            out_channels = trial.suggest_int(f"out_channels_{i}", 4, 12, step=2)
            kernel_size_mp = trial.suggest_int(f"kernel_size_{i}", 3, 4)
            kernel_size = 3

            # add layers
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size))
            layers.append(nn.ReLU())

            add_maxpool = trial.suggest_categorical(f"add_maxpool_{i}", [True, False])
            if add_maxpool and not (i==(n_clayers-1)): 
                layers.append(nn.MaxPool2d(kernel_size=kernel_size_mp, stride=3))
            # always pool the last layer
            elif i == (n_clayers-1): 
                layers.append(nn.MaxPool2d(kernel_size=kernel_size_mp, stride=3))

            # calculate size
            s_out = compute_size_out_conv2d(input_size=in_size, kernel_size=kernel_size) # size_out 
            if add_maxpool or (i==(n_clayers-1)): 
                s_out = compute_size_out_maxpool2d(input_size=s_out, kernel_size=kernel_size_mp, stride=3)   

            in_size = s_out
            in_channels = out_channels


        in_features = int((in_size ** 2) * in_channels) # use in_channels, since its eq to out_channels + remove warning

        print("In Features:", in_features)

        layers.append(nn.Flatten(start_dim=1)) # Flatten after batch
        for i in range(n_flayers):
            # add 1 dropout layer
            if i == 1: 
                dropout_rate = trial.suggest_float("fc_dropout_{}".format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p=dropout_rate))

            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(hidden_size, out_features))
        # layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective(trial):
    model = SignLangCNN(n_flayers=3,
                        n_clayers=3,
                        in_channels=3,
                        hidden_size=16,
                        in_features=128,
                        out_features=12,
                        trial=trial)

    print("Model:\n", model)
    in_tensor = torch.randn(1, 3, 128, 128)
    out = model(in_tensor)
    return 0.1

if __name__ == "__main__":
#    study = optuna.create_study(direction="maximize")
#    study.optimize(objective, n_trials=1)
#
#    print("Best trial:")
#    trial = study.best_trial
#
#    print("  Value: ", trial.value)
#
#    print("  Params: ")
#    for key, value in trial.params.items():
#        print("    {}: {}".format(key, value))
    N = 128
    sout = compute_size_out_conv2d(N,kernel_size=3)
    sout = compute_size_out_maxpool2d(sout, kernel_size=4, stride=3)
    sout = compute_size_out_conv2d(sout,kernel_size=3)
    sout = compute_size_out_maxpool2d(sout, kernel_size=3, stride=3)
    print(sout**2 * 12)


