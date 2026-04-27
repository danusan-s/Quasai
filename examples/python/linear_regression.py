import numpy as np
import pyquasai

x = pyquasai.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
y = pyquasai.tensor(np.array([[1.0], [2.0]]))
print("Input shape:", x.shape())

layer = pyquasai.Linear(2, 1)
seq = pyquasai.Sequential(
    [
        layer,
    ]
)
model = pyquasai.Model(seq)

optimizer = pyquasai.SGD(0.005, 0.9)

model.compile(pyquasai.Loss.MSE, optimizer)

model.train(x, y, 100)
