import matplotlib.pyplot as plt
import numpy as np

from mylib.particle import Particle
from mylib.cell import Cell
from mylib.prio_queue import PriorityQueue
from mylib.neigbour_search import neighbor_search_periodic  # , neigbour_search
from mylib.treebuild import build_tree
from mylib.kernel import monoghan_kernel  #, top_hat_kernel


if __name__ == "__main__":
    fig, axis = plt.subplots()

    A: np.ndarray = np.array([])
    for _ in range(1_000):
        A = np.append(A, np.array(Particle(np.random.rand(2))))

    root = Cell(
        regionLowerBound=[0.0, 0.0],
        regionHigherBound=[1.0, 1.0],
        lower_index=0,
        upper_index=len(A) - 1,
    )

    build_tree(A, root, 0)

    K = 32  # number of neighbours

    densities = []
    x_coords = []
    y_coords = []

    # for particle in A:
    #     prio_queue = PriorityQueue(K)
    #     sum_of_mass = 0
    #     sum_of_mass_monoghan = 0

    #     neighbor_search_periodic(prio_queue, root, A, particle.r, [1, 1])

    #     H = np.sqrt(prio_queue.key())
    #     for i in range(K):
    #         R = np.sqrt(-prio_queue._queue[i].key)
    #         # get the mass of each neighbours
    #         mass = prio_queue._queue[i].mass
    #         rho = mass * top_hat_kernel(R, H)
    #         # rho = mass * monoghan_kernel(R, H)
    #         sum_of_mass += rho

    #     particle.rho = sum_of_mass

    #     x_coords.append(particle.r[0])
    #     y_coords.append(particle.r[1])
    #     densities.append(particle.rho)

    ###############################################################################################
    # Parallelization
    ###############################################################################################

    def worker(particle: Particle):
        particle = particle[0]

        local_sum_of_mass = 0
        local_x_coords = []
        local_y_coords = []
        local_densities = []
        prio_queue = PriorityQueue(K)

        neighbor_search_periodic(prio_queue, root, A, particle.r, [1, 1])

        H = np.sqrt(prio_queue.key())
        for i in range(K):
            R = np.sqrt(-prio_queue._queue[i].key)
            # get the mass of each neighbours
            mass = prio_queue._queue[i].mass
            # rho = mass * top_hat_kernel(R, H)
            rho = mass * monoghan_kernel(R, H)
            local_sum_of_mass += rho

        particle.rho = local_sum_of_mass

        local_x_coords.append(particle.r[0])
        local_y_coords.append(particle.r[1])
        local_densities.append(particle.rho)

        return local_x_coords, local_y_coords, local_densities

    def my_reducer(first_tuple: ([], [], []), second_tuple: ([], [], [])):
        x_coords, y_coords, densities = first_tuple
        x_coords.extend(second_tuple[0])
        y_coords.extend(second_tuple[1])
        densities.extend(second_tuple[2])
        return x_coords, y_coords, densities

    import multiprocessing
    from functools import reduce

    chunk_size = 1
    chunks = [A[i:i + chunk_size] for i in range(0, len(A), chunk_size)]
    pool = multiprocessing.Pool()
    results = pool.map(worker, chunks)
    # unpack the final results
    x_coords, y_coords, densities = reduce(my_reducer, results)

    ###############################################################################################
    # END Parallelization
    ###############################################################################################

    plt.scatter(x_coords, y_coords, s=2, c=densities, cmap="autumn")
    plt.axis("equal")
    plt.colorbar()

    plt.show()
