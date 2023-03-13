import numpy as np
import matplotlib.pyplot as plt

from typing import Any

from particle import Particle
from cell import Cell


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

    return lagger + i + 1  # readd lower index to correct index


def build_tree(A, root: Cell, dim):
    """
    Builds a binary tree from a given root cell by partitioning a global list of particles.
    param A: global list of particles
    param root: initial cell containing all particles
    param dim: dimension to partition by
    """
    v = 0.5 * (root.regionLowerBound[dim] + root.regionHigherBound[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)

    # New cell bounds are set depending on the dimension.
    if dim == 0:
        rLow_Lower = root.regionLowerBound
        rHigh_Lower = np.array([v, root.regionHigherBound[1]])
        rLow_Upper = np.array([v, root.regionLowerBound[1]])
        rHigh_Upper = root.regionHigherBound
    else:
        rLow_Lower = root.regionLowerBound
        rHigh_Lower = np.array([root.regionHigherBound[0], v])
        rLow_Upper = np.array([root.regionLowerBound[0], v])
        rHigh_Upper = root.regionHigherBound

    if s > root.iLower:
        cLow = Cell(rLow_Lower, rHigh_Lower, root.iLower, s - 1)
        root.lowerCell = cLow
        if len(A[root.iLower : s]) > 8:
            build_tree(A, cLow, 1 - dim)

    if s <= root.iUpper:
        cHigh = Cell(rLow_Upper, rHigh_Upper, s, root.iUpper)
        root.upperCell = cHigh
        if len(A[s : root.iUpper + 1]) > 8:
            build_tree(A, cHigh, 1 - dim)


def plot_tree(axis, root: Cell, A):
    """
    Scatter points and call recursive rectangle plotter
    """
    cntr = 0
    for particle in A:
        if cntr == 0:
            axis.scatter(particle.r[0], particle.r[1], color="red", label="points")
        else:
            axis.scatter(particle.r[0], particle.r[1], color="red")
        cntr += 1
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
