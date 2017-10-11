
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