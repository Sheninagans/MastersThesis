import numpy as np

Y_array = np.array([[0.5, 2], [3, 0.112], [3.49, 6]])  # Shape (3,2)
theta = np.array([[0.5, 1.5], [2, 5.29]])  # Shape (2,2)


L_raw = np.array([
    [np.sum((Y_array[t] - theta[k, :])**2)
    for k in range(theta.shape[0])]
    for t in range(Y_array.shape[0])])
print(L_raw)

