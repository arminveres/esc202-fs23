from typing import Any
import numpy as np

# my modules
from cell import Cell
from particle import Particle
from prio_queue import PriorityQueue

# Implement the k nearest neighbor search. Use the priority queue given in the Python template and
# implement “replace” and “key” functions. Use the particle to cell distance function from the
# lecture notes or the celldist2()  given in the Python template. Are they the same?
# Optional: Also implement the ball search algorithm given in the lecture notes.


def neighbor_search_periodic(
    prio_queue: PriorityQueue,
    root: Cell,
    particles: np.ndarray[Any, Particle],
    r: np.ndarray[int, int],
    period,
):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(prio_queue, root, particles, r, rOffset)


def neighbor_search(
    prio_queue: PriorityQueue,
    root: Cell,
    particles: np.ndarray[Any, Particle],
    rParticle: np.ndarray[int, int],
    rParticleOffset: np.ndarray[int, int],
):
    """
    Do a nearest neighbor search for particle at 'r' in the tree 'root' using the priority queue
    'pq'.
    'rOffset' is the offset of the root node from unit cell, used for periodic boundaries.
    'particles' is the array of all particles.
    """
    if root is None:
        return

    ri: np.ndarray[int, int] = rParticle + rParticleOffset

    if root.lowerCell is not None and root.upperCell is not None:
        bound_lower = [
            root.lowerCell.regionHigherBound[0] - ri[0],
            root.lowerCell.regionHigherBound[1] - ri[1],
        ]
        d2_lower = dist2(root.lowerCell.rc, ri, bound_lower)
        bound_upper = [
            root.upperCell.regionHigherBound[0] - ri[0],
            root.upperCell.regionHigherBound[1] - ri[1],
        ]
        d2_upper = dist2(root.upperCell.rc, ri, bound_upper)
        # d2_lower = root.lowerCell.dist_center_to_other(ri)
        # d2_upper = root.upperCell.dist_center_to_other(ri)
        if d2_lower <= d2_upper:
            if root.lowerCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.lowerCell, particles, rParticle, rParticleOffset
                )
            if root.upperCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.upperCell, particles, rParticle, rParticleOffset
                )
        else:
            if root.upperCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.upperCell, particles, rParticle, rParticleOffset
                )
            if root.lowerCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue, root.lowerCell, particles, rParticle, rParticleOffset
                )
    elif root.lowerCell is not None:
        neighbor_search(
            prio_queue, root.lowerCell, particles, rParticle, rParticleOffset
        )
    elif root.upperCell is not None:
        neighbor_search(
            prio_queue, root.upperCell, particles, rParticle, rParticleOffset
        )
    else:  # root is a leaf cell
        # for each particle get the distance, check if smaller than the largest key and replace it
        for j in range(root.iLower, root.iUpper):
            # d2 = dist2(particles[j].r, ri)
            # d2 = particles[j].dist_center_to_other(ri)

            bound = [particles[j].r[0] - ri[0], particles[j].r[1] - ri[1]]
            d2 = dist2(particles[j].r, ri, bound)
            if d2 < prio_queue.key():
                prio_queue.replace(d2, j, particles[j].r - rParticleOffset)


def dist2(
    center_pos: np.ndarray[np.number, np.number],
    particle_pos: np.ndarray[np.number, np.number],
    bounding_length: np.ndarray[np.number, np.number],
):
    """
    Euclidian/square distance of 2 particles
    """

    distance_squared = 0
    for dimension in range(2):
        tmp = (abs(center_pos[dimension] - particle_pos[dimension]) - bounding_length[dimension])
        if tmp > 0:
            distance_squared += tmp**2
    return distance_squared

    # x = np.abs(center_pos[0] - particle_pos[0])
    # y = np.abs(center_pos[1] - particle_pos[1])
    # # Periodic boundaries
    # x = np.min(x, 1 - x)
    # y = np.min(y, 1 - y)
    # return np.sqrt(x * x + y * y)
