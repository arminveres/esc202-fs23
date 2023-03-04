import numpy as np
import matplotlib.pyplot as plt

from typing import Any

from particle import Particle
from cell import Cell


# def partition(A, i, j, v, d):
#     """
#       A: array of all particles
#       i: start index of subarray for partition
#       j: end index (inclusive) for subarray for partition
#       v: value for partition e.g. 0.5
#       d: dimension to use for partition, 0 (for x) or 1 (for y)
#     """
#     subA = A[i : j + 1]
#     start, end = 0, len(subA) - 1
#     while start <= end:
#         if subA[start].r[d] < v and subA[end].r[d] >= v:
#             start += 1; end -= 1;
#         elif subA[start].r[d] >= v and subA[end].r[d] >= v:
#             end -= 1
#         elif subA[start].r[d] < v and subA[end].r[d] < v:
#             start += 1
#         elif subA[start].r[d] >= v and subA[end].r[d] < v:
#             subA[start], subA[end] = subA[end], subA[start]
#             start += 1; end -= 1
#     return start


def partition(A: np.ndarray[Any, Particle], i: int, j: int, v: np.number, d: int):
    """
    params:
        A: array of all particles
        i: start index of subarray for partition
        j: end index (inclusive) for subarray for partition
        v: value for partition e.g. 0.5
        d: dimension to use for partition, 0 (for x) or 1 (for y)
    returns:
        s: index of right part
    """

    # case of empty particle array
    if len(A) == 0:
        return None

    interval = A[i : j + 1]
    # This index will keep track of the last 'greater than v' value
    lagger = 0
    # Keeps track of the current index
    checker = 0
    for particle in interval:
        # particle position smaller, swap needed
        if particle.r[d] < v:
            # only swap if not same index, otherwise not change made
            if lagger < checker:
                # increase the i index until a larger value is found, in order to be swappable
                while interval[lagger].r[d] < v and lagger < checker:
                    lagger += 1
                # make the swap
                interval[lagger], interval[checker] = (
                    interval[checker],
                    interval[lagger],
                )
        # increase j index
        checker += 1

    return lagger + i  # readd lower index to correct index


def build_tree(A: np.ndarray[Any, Particle], root: Cell, dim: int):
    v = 0.5 * (root.regionLowerBound[dim] + root.regionHigherBound[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)

    # early return condition
    if not s:
        return

    # check for a lower part
    # may have two parts: lower..s-1 and s..upper
    if s != 0:
        new_higher_bound = root.regionHigherBound[:]
        new_higher_bound[dim] = v
        cLow = Cell(root.regionLowerBound, new_higher_bound, root.iLower, s - 1)
        root.lowerCell = cLow
        if len(A[:s]) > 8:
            build_tree(A[:s], cLow, 1 - dim)

    # Check for an upper part
    if s <= len(A):
        new_lower_bound = root.regionLowerBound[:]
        new_lower_bound[dim] = v
        cHigh = Cell(new_lower_bound, root.regionHigherBound, 0, root.iUpper - s)
        root.upperCell = cHigh
        if len(A[s:]) > 8:
            build_tree(A[s:], cHigh, 1 - dim)


def plot_tree(root: Cell, A):
    """
    Scatter points and call recursive rectangle plotter
    """
    for particle in A:
        plt.scatter(particle.r[0], particle.r[1], color="red")
    plot_rectangles(root)


def plot_rectangles(root: Cell):
    if root.lowerCell:
        plot_rectangles(root.lowerCell)
    if root.upperCell:
        plot_rectangles(root.upperCell)
    xl = root.regionLowerBound[0]
    xh = root.regionHigherBound[0]
    yl = root.regionLowerBound[1]
    yh = root.regionHigherBound[1]
    plt.plot([xl, xh], [yl, yl], color="k")
    plt.plot([xl, xh], [yh, yh], color="k")
    plt.plot([xl, xl], [yl, yh], color="k")
    plt.plot([xh, xh], [yl, yh], color="k")
