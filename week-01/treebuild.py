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

_, gAxis = plt.subplots()

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


def buildtree(A: np.ndarray[Any, Particle], root: Cell, dim: int):
    """
    Build a Tree out a list of particles
    """

    v = 0.5 * (root.regionLowerBound[dim] + root.regionHigherBound[dim])
    # print("v: ", v)

    s = partition(A, root.iLower, root.iUpper, v, dim)
    # print("s: ", s)
    # print("adj: s: ", s)

    # may have two parts: lower..s-1 and s..upper
    if s != 0 and len(A[root.iLower : s - 1]) > 0:  # check for a lower part
        cLow = Cell(A[root.iLower].r, A[s - 1].r, root.iLower, s - 1)
        root.lowerCell = cLow
        # if there are more than 8 particles in cell:
        if len(A[root.iLower : s - 1]) >= 8:
            buildtree(A, cLow, 1 - dim)
    if len(A[s : root.iUpper]) > 0:  # check for an upper part
        cHigh = Cell(A[s].r, A[root.iUpper].r, s, root.iUpper)
        root.upperCell = cHigh
        # if there are more than 8 particles in cell:
        if len(A[s : root.iUpper]) >= 8:
            buildtree(A, cHigh, 1 - dim)


def plot_cell_tree(
    # axs: plt.Axes,
    particles: np.ndarray,
    root: Cell,
    leaf: str = "root",
    dim: int = 0,
    level: int = 0,
):
    global gAxis
    # TODO: (aver) use one single plot and plot the dividing lines onto it!
    fig, axs = plt.subplots()
    interval = particles[root.iLower : root.iUpper + 1]
    v = 0.5 * (root.regionLowerBound[dim] + root.regionHigherBound[dim])
    side = 1 if leaf == "upper" else 0

    # axs[level][side].set_title(f"At tree level: {level}, side: {leaf}")
    axs.set_title(f"At tree level: {level}, side: {leaf}")

    # set plotting limit and add padding
    # axs.set_xlim(root.regionLowerBound[0] - 1, root.regionHigherBound[0] + 1)
    # axs.set_ylim(root.regionLowerBound[1] - 1, root.regionHigherBound[1] + 1)

    for part in interval:
        axs.scatter(part.r[0], part.r[1])

    if dim == 0:
        axs.axvline(x=v, color="red")
        gAxis.axvline(x=v*(root.regionLowerBound[0]+ regionHigherBound[0]), color="red")
    elif dim == 1:
        axs.axhline(y=v, color="red")
        gAxis.axhline(y=v*(root.regionLowerBound[1]+ regionHigherBound[1]), color="red")

    if root.upperCell != None:
        plot_cell_tree(A, root.upperCell, "upper", 1 - dim, level + 1)
    if root.lowerCell != None:
        plot_cell_tree(A, root.lowerCell, "lower", 1 - dim, level + 1)


if __name__ == "__main__":

    A: np.ndarray = np.array([])
    # Build the tree
    for _ in range(20):
        A = np.append(A, np.array(Particle(np.random.rand(2))))

    regionLowerBound = np.array([0, 0])
    regionHigherBound = np.array([1, 1])
    lower = 0
    upper = A.size - 1
    root = Cell(regionLowerBound, regionHigherBound, lower, upper)
    dim = 0

    buildtree(A, root, dim)

    # fig, axs = plt.subplots(nrows=gNrSubplots, ncols=2)
    # plot_cell_tree(axs, A, root)

    for part in A:
        gAxis.scatter(part.r[0], part.r[1])

    # fig, axs = plt.subplots()
    plot_cell_tree(A, root)

    plt.show()
