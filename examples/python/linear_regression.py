import numpy as np
import pyquasai

x = pyquasai.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
print("Input shape:", x.shape())

layer = pyquasai.Linear(2, 1)
output = layer(x)
print("Output shape:", output.shape())
print("Success!")
