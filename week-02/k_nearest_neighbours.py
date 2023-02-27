from typing import Any
import numpy as np

# my modules
from cell import Cell
from particle import Particle
from heap_pq import PriorityQueue

# Implement the k nearest neighbor search. Use the priority queue given in the Python template and
# implement “replace” and “key” functions. Use the particle to cell distance function from the
# lecture notes or the celldist2()  given in the Python template. Are they the same?
# Optional: Also implement the ball search algorithm given in the lecture notes.


def neighbor_search_periodic(
    prio_queue, root: Cell, particles: np.ndarray[Any, Particle], r, period
):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(prio_queue, root, particles, r, rOffset)


def neighbor_search(
    prio_queue, root: Cell, particles: np.ndarray[Any, Particle], radius, radius_offset
):
    """
    Do a nearest neighbor search for particle at  'r' in the tree 'root'
    using the priority queue 'pq'. 'rOffset' is the offset of the root
    node from unit cell, used for periodic boundaries.
    'particles' is the array of all particles.
    """
    if root is None:
        return

    ri = radius + radius_offset
    if root.pLower is not None and root.pUpper is not None:
        d2_lower = dist2(root.pLower.rc, ri)
        d2_upper = dist2(root.pUpper.rc, ri)
        if d2_lower <= d2_upper:
            if root.pLower.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.pLower, particles, radius, radius_offset
                )
            if root.pUpper.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.pUpper, particles, radius, radius_offset
                )
        else:
            if root.pUpper.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.pUpper, particles, radius, radius_offset
                )
            if root.pLower.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.pLower, particles, radius, radius_offset
                )
    elif root.pLower is not None:
        neighbor_search(prio_queue, root.pLower, particles, radius, radius_offset)
    elif root.pUpper is not None:
        neighbor_search(prio_queue, root.pUpper, particles, radius, radius_offset)
    else:  # root is a leaf cell
        for j in range(root.iLower, root.iUpper):
            d2 = dist2(particles[j].r, ri)
            if d2 < prio_queue.key():
                prio_queue.replace(d2, j, particles[j].r - radius_offset)
    # TODO: for pq write a wrapper class that implements key() and replace() using heapq package


def dist2(pos1: int, pos2: int):
    """
    Euclidian/square distance of 2 particles
    """
    raise NotImplementedError
