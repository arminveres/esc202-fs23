import heapq

import numpy as np
from particle import Particle


class PriorityQueue:
    """
    Priority Queue using the heapq algorithm
    """

    # def __init__(self, queue_len_k: int, particle_arr: np.ndarray[Particle]):
    #     # O(N**2) Test Code
    #     # k = Number of nearest neighbors
    #     for p in particle_arr:
    #         NN = []
    #         d2NN = []
    #         for i in range(queue_len_k):
    #             d2min = float('inf')
    #             for q in particle_arr:
    #                 if p != q and q not in NN:
    #                     d2 = p.dist2(q)
    #                     if d2 < d2min:
    #                         d2min = d2
    #                         qmin = q
    #             NN.append(qmin)
    #             d2NN.append(d2min)

    #     # Here NN and d2NN lists for particle p are filled.
    #     # Compare them with the lists you got from the recursive algorithm

    def __init__(self, queue_len_k: int):
        self._queue = []
        # we instantiate the heap with infinite values
        for i in range(queue_len_k):
            sentinel = Particle([-np.inf, -np.inf], -np.inf)
            heapq.heappush(self._queue, sentinel)

    def key(self):
        """
        Return the key of highest key of an item in the priority queue
        """
        # need to reverse the minus sign, since the heapq returns the smallest item at 0 position
        return -self._queue[0].key

    def replace(self, distance: int, index: int, coordinates: np.ndarray[int, int]) -> Particle:
        """
        Replaces the head(largest) element with given 'item', and returns it
        """
        # NOTE: j==index is unused!

        particle = Particle(coordinates, -distance)
        return heapq.heapreplace(self._queue, particle)
