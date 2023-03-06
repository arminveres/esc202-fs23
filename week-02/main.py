import matplotlib.pyplot as plt
from matplotlib.patches import Circle as Pltcircle
import numpy as np

from particle import Particle
from cell import Cell
from prio_queue import PriorityQueue
from treebuild import build_tree, plot_tree
from k_nearest_neighbours import neighbor_search, neighbor_search_periodic
# from k_nearest_neighbours import neighbor_search_2

if __name__ == "__main__":
    fig, axis = plt.subplots()

    A: np.ndarray = np.array([])
    for _ in range(200):
        A = np.append(A, np.array(Particle(np.random.rand(2))))

    root = Cell(
        regionLowerBound=[0.0, 0.0],
        regionHigherBound=[1.0, 1.0],
        lower_index=0,
        upper_index=len(A) - 1,
    )

    build_tree(A, root, 0)
    plot_tree(axis, root, A)

    k = 32
    prio_queue = PriorityQueue(k)

    particle_to_search = A[0]

    # neighbor_search(prio_queue, root, A, particle_to_search.r, 0)
    neighbor_search_periodic(prio_queue, root, A, particle_to_search.r, [1, 1])

    # color the neigbours
    # axis.add_patch(Pltcircle(particle_to_search.r, prio_queue.get_max_distance(),
    #                          facecolor='yellow', edgecolor='black'))

    cntr = 0
    for part in prio_queue._queue:
        if cntr == 0:
            plt.scatter(part.r[0], part.r[1], color='g', label='k-nearest')
        else:
            plt.scatter(part.r[0], part.r[1], color='g')
        cntr += 1

    # color the particle we are looking around for
    axis.scatter(particle_to_search.r[0], particle_to_search.r[1], color='b', label='origin')
    axis.legend()
    plt.show()
