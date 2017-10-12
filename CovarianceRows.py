
# Problem: two rows of n problems, each containing m data points.
# Find the covariance for the two problems in the same row position,
# ending up with n numbers - the covariances.

import numpy as np
import pyopencl as cl

# Define...
TOL = 0.001  # Tolerance for floating point comparisons.

# Experimental data.
problems = 100
data_points = 50
row_1 = np.random.rand(problems, data_points).astype(np.float32)  # First row of problems
row_2 = np.random.rand(problems, data_points).astype(np.float32)  # Second row of problems
out_row = np.zeros(problems, dtype=np.float32)  # Output row of covariances.

print("Setting up necessities for calculating covariances.")

# Get kernel source.
kernel_source = open('covariance_rows.cl').read()

platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

print("Building program.")
program = cl.Program(context, kernel_source).build()
fc = program.find_covariances
fc.set_scalar_arg_dtypes([None, None, np.uint32, None])

queue = cl.CommandQueue(context)

# Memory stuff.
mf = cl.mem_flags
row_1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=row_1)
row_2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=row_2)
out_row_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=out_row)

fc(queue, (problems, 1), None, row_1_buf, row_2_buf, data_points, out_row_buf)

cl.enqueue_map_buffer(queue, out_row_buf, cl.map_flags.READ, 0, out_row.shape, np.float32)

print("Let us compare values!")
print("----------------------------------------------")
correct = 0
print("Calculating covariances using the numpy method.")
np_cov = np.zeros(problems)

for x in range(problems):
    # Also checking numpy value.
    np_cov[x] = np.cov([row_1[x], row_2[x]])[0, 1]


print("Comparing covariance using online OpenCL method, and Numpy method.")
for i in range(problems):
    diff_cov = out_row[i] - np_cov[i]
    if diff_cov**2 < TOL**2:
        correct += 1
    # else:
        # print("Not alike! Element", i)
        print("\n\tDiff:", diff_cov)
        print("\tOnline covariance:", out_row[i], "\tNumpy covariance:", np_cov[i])

print("Correct:", correct, "out of", problems)






