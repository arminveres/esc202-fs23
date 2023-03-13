#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


class particle:
    def __init__(self, r):
        self.r = r  # positionn of the particle [x, y]
        self.rho = 0.0  # density of the particle
        # ...  more properties of the particle


class cell:
    def __init__(self, rLow, rHigh, lower, upper):
        self.rLow = rLow  # [xMin, yMin]
        self.rHigh = rHigh  # [xMax, yMax]
        self.iLower = lower  # index to first particle in particle array
        self.iUpper = upper  # index to last particle in particle array
        self.pLower = None  # reference to tree cell for lower part
        self.pUpper = None  # reference to tree cell for upper part


def partition(A, i, j, v, d):
    """
    Input:
      A: array of all particles
      i: start index of subarray for partition
      j: end index (inclusive) for subarray for partition
      v: value for partition e.g. 0.5
      d: dimension to use for partition, 0 (for x) or 1 (for y)
    """
    # if there are no particles return None
    if len(A) == 0:
        return None

    while i < j:
        # start with i
        while A[i].r[d] < v:
            i += 1
            if i == j:
                if A[i].r[d] < v:
                    return j + 1
                else:
                    return j
        # continue with j
        while A[j].r[d] >= v:
            j -= 1
            if j == i:
                if A[i].r[d] < v:
                    return j + 1
                else:
                    return j
        # switch
        A[i].r, A[j].r = A[j].r, A[i].r


def treebuild(A, root, dim):
    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)
    if s:
        # may have two parts: lower..s-1 and s..upper
        if s != 0:  # there is a lower part
            new_rHigh = root.rHigh[:]
            new_rHigh[dim] = v
            cLow = cell(root.rLow, new_rHigh, root.iLower, s - 1)
            root.pLower = cLow
            if len(A[:s]) > 8:
                treebuild(A[:s], cLow, 1 - dim)
        if s <= len(A):  # there is an upper par
            new_rLow = root.rLow[:]
            new_rLow[dim] = v
            cHigh = cell(new_rLow, root.rHigh, 0, root.iUpper - s)
            root.pUpper = cHigh
            if len(A[s:]) > 8:  # there are more than 8 particles in cell
                treebuild(A[s:], cHigh, 1 - dim)


def plottree(root: cell):
    # draw a rectangle specified by rLow and rHigh
    if root.pLower:
        plottree(root.pLower)
    if root.pUpper:
        plottree(root.pUpper)
    xl = root.rLow[0]
    xh = root.rHigh[0]
    yl = root.rLow[1]
    yh = root.rHigh[1]
    print(xl, xh, yl, yh)
    plt.plot([xl, xh], [yl, yl], color="k")
    plt.plot([xl, xh], [yh, yh], color="k")
    plt.plot([xl, xl], [yl, yh], color="k")
    plt.plot([xh, xh], [yl, yh], color="k")


def plot_tree(root: cell):
    """
    Scatter points and call recursive rectangle plotter
    """
    for particle in A:
        plt.scatter(particle.r[0], particle.r[1], color="red")
    plot_rectangles(root)


def plot_rectangles(root: cell):
    if root.pLower:
        plot_tree(root.pLower)
    if root.pUpper:
        plot_tree(root.pUpper)
    xl = root.rLow[0]
    xh = root.rHigh[0]
    yl = root.rLow[1]
    yh = root.rHigh[1]
    plt.plot([xl, xh], [yl, yl], color="k")
    plt.plot([xl, xh], [yh, yh], color="k")
    plt.plot([xl, xl], [yl, yh], color="k")
    plt.plot([xh, xh], [yl, yh], color="k")


if __name__ == "__main__":
    import random

    particles = []
    for k in range(200):
        particles.append(
            # particle(np.array([random.uniform(0.0, 100.0), random.uniform(0.0, 100.0)]))
            particle(np.array([random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]))
        )
    A = np.array(particles)

    root = cell([0.0, 0.0], [1.0, 1.0], 0, len(A) - 1)
    # root = cell([0, 0], [100, 100], 0, len(A) - 1)

    treebuild(A, root, dim=0)

    for ele in A:
        plt.scatter(ele.r[0], ele.r[1], color="red")
    plottree(root)
    # plot_tree(root)

    plt.show()
