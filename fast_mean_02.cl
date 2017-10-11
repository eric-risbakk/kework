

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

    // Starts everything by setting indices for input first.
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