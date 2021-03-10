
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr

from convexnmf import ConvexNMF

start_date = datetime(2019, 1, 1)
end_date = datetime(2020, 10, 10)

symbol_list = ['BAC', 'BK', 'C', 'CFG',
                'CMA', 'COF', 'FITB', 'GS',
                'JPM', 'KEY', 'MS',
                'MTB', 'PBCT', 'PNC', 
                'STT', 'USB', 'WFC', 'ZION']

src_data = pdr.data.DataReader(symbol_list, 'yahoo', start_date, end_date)

closing = src_data["Adj Close"].to_numpy()
closing_normalized = closing / np.linalg.norm(closing, axis = 0)

model = ConvexNMF()
model.report_progress = False
model.max_iterations  = 5000

regularization_path_values = []
for pow_ten in range(2,-3, -1):
    for coeff in range(9, 0,-1):
        reg_value = coeff*(10.0**pow_ten)
        regularization_path_values.append(reg_value)

row_norm = lambda array: np.linalg.norm(array, axis = 1)
objective = lambda W: 0.5*np.linalg.norm(np.dot(closing_normalized,W)-closing_normalized)**2
sparsity  = lambda W: np.sum(np.linalg.norm(W, axis =1 ))

sparsity_path = np.zeros((len(regularization_path_values),1))
objective_path = np.zeros_like(sparsity_path)

min_norm_value = 1.0e4
min_norm_index = 0
optimal_reg = 0.0

for (index, value) in enumerate(regularization_path_values):
    result = model.Fit(closing_normalized, value)
    sparsity_path[index]= objective(result.W)
    objective_path[index] = sparsity(result.W)

    point = np.array([sparsity_path[index], objective_path[index]])
    if np.linalg.norm(point) <= min_norm_value:
        min_norm_value = np.linalg.norm(point)
        optimal_reg = value
        min_norm_index = index

plt.figure(0)
plt.scatter(sparsity_path, objective_path)
plt.scatter(sparsity_path[min_norm_index], objective_path[min_norm_index], marker=(5,2))
plt.show()

result = model.Fit(closing_normalized, optimal_reg)
plt.figure(1)
plt.imshow(result.W)
plt.show()