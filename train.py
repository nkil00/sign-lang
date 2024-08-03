import torch
from preprocessing import preprocessing
import getopt
import sys

TRAIN_SIZE = .8
BATCH_SIZE = 30
LEARNING_RATE = 0.001
EPOCHS = 10

# read in hyper_params
args_list = sys.argv[1:]
opts = "t:b:e:l:"
lopts = ["train_size=", "batch_size=", "epochs=", "learning_rate="]

try:
    args, vals = getopt.getopt(args_list, opts, lopts)

    for ca, cv in args:
        if ca in ("-t", "--train_size"):
            TRAIN_SIZE = float(cv)
        if ca in ("-l", "--learning_rate"):
            LEARNING_RATE = float(cv)
        if ca in ("-e", "--epochs"):
            EPOCHS = int(cv)
        if ca in ("-b", "--batch_size"):
            BATCH_SIZE = int(cv)



except getopt.error as e:
    print(e)
    exit()

print("EPOCHS:", EPOCHS)
print("BATCH_SIZE:", BATCH_SIZE)
print("LEARNING RATE:", LEARNING_RATE)
print("TRAIN SIZE:", TRAIN_SIZE)

# get test and data loader
test_loader, train_loader = preprocessing.get_test_train_loader()
print(test_loader, train_loader)


