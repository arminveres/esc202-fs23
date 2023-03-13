import numpy as np
import matplotlib.pyplot as plt

from typing import Any

"""
Instructions

Build a binary tree of cells using the partitioning of particles function we introduced in class.

The hard part is making sure your partition function is really bomb proof, check all “edge cases”
(e.g., no particles in the cell, all particles on one side or other of the partition, already
partitioned data, particles in the inverted order of the partition, etc…). Write a boolean test
functions for each of these cases. Call all test functions in sequence and check if they all
succeed.
Once you have this, then recursively partition the partitions and build cells linked into a tree as
you go. Partition alternately in x and y dimensions, or simply partition the longest dimension of
the given cell.
"""


class Particle:
    def __init__(self, r: np.ndarray):
        self.r: np.ndarray = r  # positionn of the particle [x, y]
        self.rho = 0.0  # density of the particle
        # ...  more properties of the particle


class Cell:
    def __init__(
        self,
        regionLowerBound: np.ndarray,
        regionHigherBound: np.ndarray,
        lower_index: int,
        upper_index: int,
    ):
        self.regionLowerBound = regionLowerBound  # [xMin, yMin]
        self.regionHigherBound = regionHigherBound  # [xMax, yMax]
        self.iLower = lower_index  # index to first particle in particle array
        self.iUpper = upper_index  # index to last particle in particle array
        self.lowerCell = None  # reference to tree cell for lower part
        self.upperCell = None  # reference to tree cell for upper part


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


def build_tree(A: np.ndarray, root: Cell, dim: int):
    """
    Builds a binary tree from a given root cell by partitioning a global list of particles.
    :param A: global list of particles
    :param root: initial cell containing all particles
    :param dim: dimension to partition by
    :return:
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

    # The left cell is generated if a left partition exists and the branching continued.
    if s > root.iLower:
        cLow = Cell(rLow_Lower, rHigh_Lower, root.iLower, s - 1)
        root.lowerCell = cLow
        if len(A[root.iLower:s]) > 8:
            build_tree(A, cLow, 1 - dim)

    # The right cell is generated if a right partition exists and the branching continued.
    if s <= root.iUpper:
        cHigh = Cell(rLow_Upper, rHigh_Upper, s, root.iUpper)
        root.upperCell = cHigh
        if len(A[s:root.iUpper + 1]) > 8:
            build_tree(A, cHigh, 1 - dim)


def plot_tree(root: Cell):
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


if __name__ == "__main__":
    A: np.ndarray = np.array([])
    for _ in range(100):
        A = np.append(A, np.array(Particle(np.random.rand(2))))

    root = Cell(
        regionLowerBound=[0.0, 0.0],
        regionHigherBound=[1.0, 1.0],
        lower_index=0,
        upper_index=len(A) - 1,
    )
    build_tree(A, root, 0)

    plot_tree(root)
    plt.show()
