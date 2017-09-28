
import pyopencl as cl
import numpy as np

# TODO: Lol, a function has only one output by definition.

# Experimental data.
problems = 10  # Let a row correspond to a problem.
data_points = 10
inp = np.random.rand(problems, data_points).astype(np.float32)
goal = np.empty(problems)
out_mean = np.empty_like(goal)
out_std = np.empty_like(goal)

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
                value = input[gid+i];
                delta = value - mean;
                mean += delta/(i+1);
                variance += delta*(value-mean);
            }

            variance /= (length - 1);

            meanOut[gid] = mean;
            stdOut[gid] = sqrt(variance);
        }    
"""

platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

program = cl.Program(context, kernel_source).build()
program.fast_mean_test.set_scalar_arg_dtypes([None, np.uint32, None, None])

queue = cl.CommandQueue(context)

mem_flags = cl.mem_flags

inp_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=inp)
mean_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out_mean.nbytes)
std_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, out_std.nbytes)

program.fast_mean_test(queue, (2, problems), None, inp_buf, data_points, mean_buf, std_buf)

cl.enqueue_copy(queue, out_mean, mean_buf)
cl.enqueue_copy(queue, out_std, std_buf)

print(out_mean)
print(out_std)

