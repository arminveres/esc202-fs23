from typing import Any
import numpy as np
from numpy.typing import NDArray

# my modules
from mylib.cell import Cell
from mylib.particle import Particle, PriorityQueue

# Implement the k nearest neighbor search. Use the priority queue given in the Python template and
# implement “replace” and “key” functions. Use the particle to cell distance function from the
# lecture notes or the celldist2()  given in the Python template. Are they the same?
# Optional: Also implement the ball search algorithm given in the lecture notes.


def neighbor_search_periodic(
    prio_queue: PriorityQueue,
    root: Cell,
    particles: list[Particle],
    r: NDArray[np.float64],
    period: NDArray[np.float64]
):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(prio_queue, root, particles, r, rOffset)


def neighbor_search(
    prio_queue: PriorityQueue,
    root: Cell,
    particles: list[Particle],
    rootParticle: NDArray[np.float64],
    rootParticleOffset: NDArray[np.float64],
):
    """
    Do a nearest neighbor search for particle at 'r' in the tree 'root' using the priority queue
    'pq'.
    'rOffset' is the offset of the root node from unit cell, used for periodic boundaries.
    'particles' is the array of all particles.
    """
    if root is None:
        return

    ri: NDArray[np.float64] = rootParticle + rootParticleOffset

    if root.lowerCell is not None and root.upperCell is not None:
        d2_lower = dist2(root.lowerCell.rc, ri)
        d2_upper = dist2(root.upperCell.rc, ri)
        if d2_lower <= d2_upper:
            if root.lowerCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue,
                    root.lowerCell,
                    particles,
                    rootParticle,
                    rootParticleOffset,
                )
            if root.upperCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue,
                    root.upperCell,
                    particles,
                    rootParticle,
                    rootParticleOffset,
                )
        else:
            if root.upperCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue,
                    root.upperCell,
                    particles,
                    rootParticle,
                    rootParticleOffset,
                )
            if root.lowerCell.celldist2(ri) < prio_queue.key():
                neighbor_search(
                    prio_queue,
                    root.lowerCell,
                    particles,
                    rootParticle,
                    rootParticleOffset,
                )
    elif root.lowerCell is not None:
        neighbor_search(
            prio_queue, root.lowerCell, particles, rootParticle, rootParticleOffset
        )
    elif root.upperCell is not None:
        neighbor_search(
            prio_queue, root.upperCell, particles, rootParticle, rootParticleOffset
        )
    else:  # root is a leaf cell
        for part in particles[root.iLower: root.iUpper + 1]:
            distance_squared = dist2(part.r, ri)
            if distance_squared < prio_queue.key() and distance_squared != 0.0:
                part.key = -distance_squared
                prio_queue.replace(part)


def dist2(
    center_pos: NDArray[np.float64],
    particle_pos: NDArray[np.float64]
):
    """
    Euclidian/square distance of 2 particles
    """
    x1, y1 = center_pos
    x2, y2 = particle_pos
    return abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2
