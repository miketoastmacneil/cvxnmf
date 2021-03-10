
import matplotlib.pyplot as plt
import numpy as np

from convexnmf import ConvexNMF

model = ConvexNMF()
model.report_progress = True
model.max_iterations = 5000

A = np.array([[1, 0.1, 0, 0.1, 0, 0.15, 0.04, 0.2, 0.1],
              [0, 0.3, 0, 0.2, 1, 0.1,  0.4, 0.56, 0.3],
              [0, 0.4, 1, .3, 0, 0.69, 0.67, 0.33, 0.12]], dtype = np.float64)

result = model.Fit(A, 0.9)

recon = np.dot(A, result.W) - A

print(np.linalg.norm(result.W, axis = 1))
plt.imshow(result.W)
plt.show()

regularization_path_values = []
for pow_ten in range(2,-3, -1):
    for coeff in range(9, 0,-1):
        reg_value = coeff*(10.0**pow_ten)
        regularization_path_values.append(reg_value)

model.report_progress = False
objective = lambda W: 0.5*np.linalg.norm(np.dot(A,W)-A)**2
sparsity  = lambda W: np.sum(np.linalg.norm(W, axis =1 ))

sparsity_path = np.zeros((len(regularization_path_values),1))
objective_path = np.zeros_like(sparsity_path)

min_norm_value = 1.0e4
min_norm_index = 0
optimal_reg = 0.0

for (index, value) in enumerate(regularization_path_values):
    result = model.Fit(A, value)
    sparsity_path[index]= objective(result.W)
    objective_path[index] = sparsity(result.W)

    point = np.array([sparsity_path[index], objective_path[index]])
    if np.linalg.norm(point) <= min_norm_value:
        min_norm_value = np.linalg.norm(point)
        optimal_reg = value
        min_norm_index = index

plt.scatter(sparsity_path, objective_path)
plt.scatter(sparsity_path[min_norm_index], objective_path[min_norm_index], marker=(5,2))
plt.show()