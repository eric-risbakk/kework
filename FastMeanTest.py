
import pyopencl as cl
import numpy as np

# Experimental data.
problems = np.float32(1000)
data_points = np.float32(1000)
inp = np.random.rand(problems, data_points).astype(np.float32)
out_mean = np.empty_like(problems)
out_std = np.empty_like(problems)

# The program. Finds the mean and standard deviance of the data from input.
kernel_source = """
    __kernel void fast_mean_test(
        __global float* input,
        uint length,
        __global float* meanOut
        __global float* stdOut)
        {
            float mean = 0;
            float variance = 0;
            float delta;
            float value;

            for (uint i = 0; i < length; ++i) {
                value = input[i]
                delta = value - mean;
                mean += delta/(i+1);
                variance += delta*(value-mean)
            }

            variance /= (length - 1);

            *meanOut = mean;
            stdOut = sqrt(variance)
        }    
"""

platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

program = cl.Program(context, kernel_source).build()

queue = cl.CommandQueue(context)

mem_flags = cl.mem_flags

inp_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=inp)
mean_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out_mean.nbytes)
std_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out_std.nbytes)

program.fast_mean_test(queue, out_mean.shape, None, inp_buf, data_points, mean_buf, std_buf)

cl.enqueue_copy(queue, out_mean, mean_buf)
cl.enqueue_copy(queue, out_std, std_buf)

