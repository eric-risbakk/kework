
import scipy.stats as spstat
import numpy as np
import numpy.random as npr
import scipy.stats as spstat
import numpy as np
import numpy.random as npr

__author__ = 'Eric Risbakk'
__date__ = "2017-09-14"
__maintainer__ = "Eric Risbakk"
__email__ = "e.risbakk@student.maastrichtuniversity.nl"

DEBUG = False
TEST = True

# Push test.


def online_mean_3d(ndarray, axis):
    """
    Finds the mean in a 3d ndarray along the specified (int) axis.

    Takes in the already complete array and finds the mean for it.

    :param axis: Axis which we find the mean on.
    :param ndarray: A 1-dimensional array.
    :return: Arithmetic mean of simpleArray
    """

    if len(ndarray) < 2:
        return ndarray
    else:

        """
        # Creating the ndarray which will be the mean.
        dimensions = []
        for i in range(ndarray.ndim):
            if i == axis:
                continue
            dimensions.append(ndarray.shape[i])
        m = np.zeros(dimensions)

        # Getting the mean for all points, using some recursion!
        tempAxis = 0

         """

        mean = ndarray[0]
        n = 1
        for x in ndarray[1:]:
            mean = online_mean_step(x, mean, n)
            n += 1
        return mean

def recursive_truncation(mean, ndarray, currentAxis, axis):
    # End-statement.
    if currentAxis == ndarray.ndim:
        return
    # Skip this.
    if currentAxis == axis:
        recursive_truncation(mean, ndarray, currentAxis + 1, axis)

    # Let's go depth first!
    # Let's truncate this axis.
    # TODO: FIGURE THIS OUT. IS IT EVEN POSSIBLE?
    # TODO: MAYBE I SHOULD BE USING A TUPLE OR SOMETHING.
    for i in range(ndarray.shape[axis]):





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
            mean[i, j] = online_mean_3d(a1[i, j, :])

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

if TEST:
    # First is rows, second is columns
    a1 = npr.rand(2, 2, 2)
    a1 *= 10
    array_mean0 = np.mean(a1, axis=0)
    array_mean1 = np.mean(a1, axis=1)
    array_mean2 = np.mean(a1, axis=2)

    print(a1)
    print("\nMeans:")

    print("\n First axis:")
    print(array_mean0)
    print("\n Second axis:")
    print(array_mean1)
    print("\n Third axis:")
    print(array_mean2)

    print("\nLet us attempt using axisMeanCheck.")
    a_mean = axis_mean_check(a1)
    print(a_mean)

    print("\nLet us attempt using axisOnlineMeanCheck.")
    b_mean = axis_online_mean_check(a1)
    print(b_mean)
    print("Finished.")

    # End.
