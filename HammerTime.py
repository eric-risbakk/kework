
import numpy as np

# Experimental data.
problems = 10
data_points = 2
inp = np.random.rand(problems, data_points).astype(np.float32)
out_mean = np.empty_like(problems)
out_std = np.empty_like(problems)

print(inp)
