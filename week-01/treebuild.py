import numpy as np

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
    def __init__(self, r):
        self.r = r  # positionn of the particle [x, y]
        self.rho = 0.0  # density of the particle
        # ...  more properties of the particle


class Cell:
    def __init__(self, rLow, rHigh, lower, upper):
        self.rLow = rLow  # [xMin, yMin]
        self.rHigh = rHigh  # [yMax, yMax]
        self.iLower = lower  # index to first particle in particle array
        self.iUpper = upper  # index to last particle in particle array
        self.pLower = None  # reference to tree cell for lower part
        self.pUpper = None  # reference to tree cell for upper part


def partition(A: np.array, i: int, j: int, v: np.number, d: int):
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
    i_idx = 0
    # Keeps track of the current index
    j_idx = 0
    for particle in interval:
        # particle position smaller, swap needed
        if particle.r[d] < v:
            # only swap if not same index, otherwise not change made
            if i_idx < j_idx:
                # increase the i index until a larger value is found, in order to be swappable
                while interval[i_idx].r[d] < v and i_idx < j_idx:
                    i_idx += 1
                # make the swap
                interval[i_idx], interval[j_idx] = interval[j_idx], interval[i_idx]
        # increase j index
        j_idx += 1

    # for particle in A:
    #     print(particle.r[d])
    # print(f"i: {i_idx}, j: {j_idx}")

    return i_idx  # return s


def treebuild(A: np.ndarray[Particle], root: Cell, dim: int):
    """
    Build a Tree out a list of particles
    """
    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)
    # may have two parts: lower..s-1 and s..upper
    # if there is a lower part:
    if len(A[root.iLower : s - 1]) > 0:
        #  TODO: (aver) Correctly instantiale Cells in recursive call
        cLow = Cell(A[root.iLower].r, A[s - 1].r, root.iLower, s - 1)
        root.pLower = cLow
        # if there are more than 8 particles in cell:
        if len(A[root.iLower : s - 1]) >= 8:
            treebuild(A, cLow, 1 - dim)
    # if there is an upper part:
    if len(A[s : root.iUpper]) > 0:
        cHigh = Cell(A[s].r, A[root.iUpper].r, s, root.iUpper)
        root.pUpper = cHigh
        # if there are more than 8 particles in cell:
        if len(A[s : root.iUpper]) >= 8:
            treebuild(A, cHigh, 1 - dim)
    # graphical representation of tree


def plottree(root: Cell):
    #  TODO: (aver) implement plotting function
    pass


# Tests
#  TODO: (aver) implement test cases


def test1() -> bool:
    # A = initialize with particle with sequential coordinates in x, same yMax
    A = None
    s = partition(A, 0, 10, 0.5, 0)
    return s == 5


def test2() -> bool:
    return False


def run_all_tests() -> bool:
    if not test1():
        return False
    return True


# add other testcases

if __name__ == "__main__":
    r = np.array([0.8, 1.0])
    p = Particle(r)
    r = np.array([0.6, 0.9])
    p2 = Particle(r)
    r = np.array([0.2, 0.8])
    p3 = Particle(r)
    r = np.array([0.3, 0.8])
    p4 = Particle(r)
    r = np.array([0.3, 0.7])
    p5 = Particle(r)
    r = np.array([0.4, 0.7])
    p6 = Particle(r)
    r = np.array([0.9, 0.6])
    p7 = Particle(r)
    r = np.array([0.5, 0.6])
    p8 = Particle(r)
    r = np.array([0.3, 0.5])
    p9 = Particle(r)
    r = np.array([0.5, 0.4])
    p10 = Particle(r)

    # Create array A with particles
    A = np.array([p, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    # Build the tree

    rLow = np.array([0, 0])
    rHigh = np.array([1, 1])
    lower = 0
    upper = A.size - 1
    root = Cell(rLow, rHigh, lower, upper)
    dim = 0

    # # before
    # A_len = len(A) - 1
    # print(A_len)
    # for particle in A:
    #     print(particle.r[1])
    # print("\n")
    # partition(A, 0, A_len, 0.8, 1)
    # print("\n")
    # # after
    # for particle in A:
    #     print(particle.r[1])

    # treebuild(A, root, dim)
    # plottree(root)
