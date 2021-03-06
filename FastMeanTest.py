
import pyopencl as cl
import numpy as np

# Define
TOL = 0.001  # Tolerance for floating point type comparisons

# Experimental data.
problems = 100  # Let a row correspond to a problem.
data_points = 50  # Elements in each row.
inp = np.random.rand(problems, data_points).astype(np.float32)  # input.
out_mean = np.zeros(problems, dtype=np.float32)  # Output
out_std = np.zeros(problems, dtype=np.float32)  # Output

# Get source code.
kernel_source = open('fast_mean_02.cl').read()

platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

print("Building program.")
program = cl.Program(context, kernel_source).build()
fmt = program.fast_mean_test
fmt.set_scalar_arg_dtypes([None, np.uint32, None, None])

queue = cl.CommandQueue(context)

mem_flags = cl.mem_flags

inp_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=inp)
# mean_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out_mean.nbytes)
# std_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out_std.nbytes)
mean_buf = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.USE_HOST_PTR, hostbuf=out_mean)
std_buf = cl.Buffer(context, mem_flags.WRITE_ONLY | mem_flags.USE_HOST_PTR, hostbuf=out_std)

fmt(queue, inp.shape, None, inp_buf, data_points, mean_buf, std_buf)

# Replacing the enqueue_copy with enqueue_map_buffer.
# cl.enqueue_copy(queue, out_mean, mean_buf)
# cl.enqueue_copy(queue, out_std, std_buf)
# enqueue_map_buffer(queue, buf, flags, offset, shape, dtype, order="C", strides=None, wait_for=None, is_blocking=True)
cl.enqueue_map_buffer(queue,
                      buf=mean_buf,
                      flags=cl.map_flags.READ,
                      offset=0,
                      shape=out_mean.shape,
                      dtype=np.float32)
cl.enqueue_map_buffer(queue,
                      buf=std_buf,
                      flags=cl.map_flags.READ,
                      offset=0,
                      shape=out_std.shape,
                      dtype=np.float32)

print("Let us compare values!")
print("------------------------------------------------------")
means = np.mean(inp, axis=1)
stds = np.std(inp, axis=1, ddof=1)
correct = 0
for i in range(problems):
    diff_m = out_mean[i] - means[i]
    diff_s = out_std[i] - stds[i]
    if (diff_m**2 < TOL**2) & (diff_s**2 < TOL**2):
        correct += 1
    # else:
        # print("Not alike! Element:", i)
        print("\tdiff_m", diff_m, "\tdiff_s", diff_s)
        print("\tmeans/out_mean (", means[i], "/", out_mean[i], ")")
        print("\tstds/out_std (", stds[i], "/", out_std[i])

print("Correct:", correct, "out of ", problems, ".")

