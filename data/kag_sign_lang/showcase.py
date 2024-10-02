import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SIGN_LANG_TRAIN_PATH = "./sign_mnist_test.csv"

df = pd.read_csv(SIGN_LANG_TRAIN_PATH)
df.head(2)
HEIGHT, WIDTH = 28, 28
im = np.zeros((HEIGHT, WIDTH))
df.loc(0)[0][1]
idx = 3
step = WIDTH
start = 1
end = step + 1
for i in range(HEIGHT):
    im[i] = df.iloc[idx][start:end]
    # print(start, end)
    start += step
    end += step
plt.imshow(im, cmap="gray")
label = int(df.iloc[idx][0])
plt.title(f"Label: {label}")
plt.show()

