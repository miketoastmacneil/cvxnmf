

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr

from convexnmf import ConvexNMF


model = ConvexNMF()
model.report_progress = True
model.max_iterations = 5000
model.iteration_to_print = 1

A = 8.0*np.random.randn(6000,6000)
A = np.maximum(A, 0.0)

result = model.Fit(A, 0.9)
