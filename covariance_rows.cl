
// File for computing the covariance between two rows.

// This calculates the covariance using an online version.
// See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance
// Indexing is already set from find_covariances.
void online_covariance(
    __global float* row_1,
    __global float* row_2,
    uint length,
    __global float* row_out
)
{
    float meanx = 0;
    float meany = 0;
    float c = 0;
    int n = 0;
    float dx;

    for (int x = 0; x < length; ++x)
    {
        for (int y = 0; y < length; ++y)
        {
            dx = 0;
            n += 1;
            dx = row_1[x] - meanx;
            meanx += dx / n
            meany += (row_2[y] - meany)/n
            c += dx*(row_2[y] - meany)
        }
    }

    *row_out = c/n;
}

__kernel void find_covariances(
    __global float* row_1,
    __global float* row_2,
    uint length,
    __global float* row_out
)
{
    // Set indices.
    gid = get_global_id(0);
    local float* inp_1_loc = row_1 + gid*length;
    local float* inp_2_loc = row_2 + gid*length;
    online_covariance(inp_1_loc, inp_2_loc, length, &row_out[gid]);
}