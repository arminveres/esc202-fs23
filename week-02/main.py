import matplotlib.pyplot as plt
import numpy as np

from particle import Particle
from cell import Cell
from prio_queue import PriorityQueue
from treebuild import build_tree, plot_tree
from k_nearest_neighbours import neighbor_search

if __name__ == "__main__":

    A: np.ndarray = np.array([])
    for _ in range(50):
        A = np.append(A, np.array(Particle(np.random.rand(2))))

    root = Cell(
        regionLowerBound=[0.0, 0.0],
        regionHigherBound=[1.0, 1.0],
        lower_index=0,
        upper_index=len(A) - 1,
    )

    build_tree(A, root, 0)
    plot_tree(root, A)

    prio_queue = PriorityQueue(32)

    particle_to_search = A[0]

    neighbor_search(prio_queue, root, A, particle_to_search.r, 0)

    for part in prio_queue._queue:
        print(part)

    for part in prio_queue._queue:
        # print(f"Key: {part.key}, coords: {part.r}")
        plt.scatter(part.r[0], part.r[1], color='g')

    # color the particle we are looking around for
    plt.scatter(particle_to_search.r[0], particle_to_search.r[1], color='b')

    plt.show()
