
import scipy.stats as spstat
import numpy as np
import numpy.random as npr

# From Robbert:
# As input you have a 3d ndarray, where we would like to take the mean across the last axis:
# numpy.mean(  my_data,   axis=2)
# You can simulate it with some random data of 3d shape.

__author__ = 'Eric Risbakk'
__date__ = "2017-09-14"
__maintainer__ = "Eric Risbakk"
__email__ = "e.risbakk@student.maastrichtuniversity.nl"

DEBUG = False


def online_mean_check(simple_array):
    """
    Meant to confirm that the online mean method works.

    Takes in the already complete array and finds the mean for it.

    :param simple_array: A 1-dimensional array.
    :return: Arithmetic mean of simpleArray
    """
    if len(simple_array) < 2:
        return simple_array
    else:
        mean = simple_array[0]
        n = 1
        for x in simple_array[1:]:
            mean = online_mean_step(x, mean, n)
            n += 1
        return mean


def online_mean_step(new_element, mean, n):
    """
    Updates the mean, given newElement, old mean, and the number of elements before we add newElement.

    NB: This method does not increase the number of element.

    :param new_element: New Element.
    :param mean: Old mean.
    :param n: Old number of elements.
    :return: The new mean.
    """
    return (mean * n + new_element) / (n + 1)


def axis_mean_check(ndarray):
    """
    Checks the mean of the last axis of a 3d ndarray, using the regular rule for average.
    :param ndarray: 3d ndarray
    :return: a 2d ndarray average in the direction of the last axis.
    """
    if DEBUG:
        print("axisMeanCheck begun.")
    x = ndarray.shape[0]
    y = ndarray.shape[1]
    z = ndarray.shape[2]
    if DEBUG:
        print("Dimensions: ({} {} {})".format(x, y, z))

    mean = np.zeros((x, y))

    if DEBUG:
        print("ndarray of zeroes created.")
    if DEBUG:
        print("Dimensions: ({} {})".format(mean.shape[0], mean.shape[1]))

    if DEBUG:
        print(mean)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                if DEBUG:
                    print("Dim({} {} {})".format(i, j, k))
                mean[i, j] += ndarray[i, j, k]

    if DEBUG:
        print("Collapsed axis.")
    if DEBUG:
        print(mean)

    return mean/z


def axis_online_mean_check(a1):
    """
    Checks the mean of the last axis of a 3d ndarray, using onlineMeanCheck.

    :param a1: The 3d ndarray.
    :return: 2d ndarray averaged.
    """
    if DEBUG:
        print("axisOnlineMeanCheck begun.")
    x = a1.shape[0]
    y = a1.shape[1]
    z = a1.shape[2]
    if DEBUG:
        print("Dimensions: ({} {} {})".format(x, y, z))

    mean = np.zeros((x, y))

    if DEBUG:
        print("ndarray of zeroes created.")
    if DEBUG:
        print("Dimensions: ({} {})".format(mean.shape[0], mean.shape[1]))

    if DEBUG:
        print(mean)

    for i in range(x):
        for j in range(y):
            mean[i, j] = online_mean_check(a1[i, j, :])

    if DEBUG:
        print("End axisOnlineMeanCheck")
    if DEBUG:
        print(mean)

    return mean


def get_avg(simple_array):
    mean = 0
    for x in simple_array:
        mean += x

    return mean/len(simple_array)


# First is rows, second is columns
array = npr.rand(2, 2, 2)
array *= 10
array_mean0 = np.mean(array, axis=0)
array_mean1 = np.mean(array, axis=1)
array_mean2 = np.mean(array, axis=2)

print(array)
print("\nMeans:")

print("\n First axis:")
print(array_mean0)
print("\n Second axis:")
print(array_mean1)
print("\n Third axis:")
print(array_mean2)

print("\nLet us attempt using axisMeanCheck.")
a_mean = axis_mean_check(array)
print(a_mean)

print("\nLet us attempt using axisOnlineMeanCheck.")
b_mean = axis_online_mean_check(array)
print(b_mean)
print("Finished.")

# End.
