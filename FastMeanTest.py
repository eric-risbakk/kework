
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

# The program. Finds the mean and standard deviance of the data from input.
kernel_source = """
    __kernel void fast_mean_test(
        __global float* input,
        uint length,
        __global float* meanOut,
        __global float* stdOut)
        {
            int gid = get_global_id(0);
            float mean = 0;
            float variance = 0;
            float delta;
            float value;

            for (uint i = 0; i < length; ++i) {
                value = input[i + gid*length];
                delta = value - mean;
                mean += delta/(i+1);
                variance += delta*(value-mean);
            }

            variance /= (length - 1);

            meanOut[gid] = mean;
            stdOut[gid] = sqrt(variance);
        }    
"""

kernel_source_fun = """
    // Calculates mean and variance with index already being set in input.
    // The meanOut and stdOut using get_global_id() to find their spot has been
    // abstracted out Now it accommodates (or so I think) the memory address sent.
    void mean_fancy_indexing(
        __global float* input,
        uint length,
        __global float* meanOut,
        __global float* stdOut) 
    {
        float mean = 0;
        float variance = 0;
        float delta;
        float value;

        for (uint i = 0; i < length; ++i) {
            value = input[i];
            delta = value - mean;
            mean += delta/(i+1);
            variance += delta*(value-mean);
        }

        variance /= (length - 1);

        *meanOut = mean;
        *stdOut = sqrt(variance);
    }

    __kernel void fast_mean_test(
        __global float* input,
        uint length,
        __global float* meanOut,
        __global float* stdOut)
        {
            int gid = get_global_id(0);
            global float* inp_loc = input + gid*length;
            mean_fancy_indexing(inp_loc, length, &meanOut[gid], &stdOut[gid]);
        }
"""

platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

print("Building program.")
program = cl.Program(context, kernel_source_fun).build()
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
    else:
        print("Not alike! Element:", i)
        print("\tdiff_m", diff_m, "\tdiff_s", diff_s)
        print("\tmeans/out_mean (", means[i], "/", out_mean[i], ")")
        print("\tstds/out_std (", stds[i], "/", out_std[i])

print("Correct:", correct, "out of ", problems, ".")

