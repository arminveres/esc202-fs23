import heapq as hq
import numpy as np

from mylib.particle import Particle


class PriorityQueue:
    """
    Priority Queue using the heapq algorithm
    """

    def __init__(self, queue_len_k: int):
        self._queue = []
        # we instantiate the heap with infinite values
        for _ in range(queue_len_k):
            sentinel = Particle([-np.inf, -np.inf], -np.inf)
            hq.heappush(self._queue, sentinel)

    def __repr__(self):
        return str(self._queue)

    def key(self):
        """
        Return the key of highest key of an item in the priority queue
        """
        # need to reverse the minus sign, since the heapq returns the smallest item at 0 position
        return -self._queue[0].key

    def replace(self, distance: int, coordinates: np.ndarray[int, int]) -> Particle:
        """
        Replaces the head(largest) element with given 'item', and returns it
        """
        particle = Particle(coordinates, -distance)
        return hq.heapreplace(self._queue, particle)

    def get_max_distance(self):
        return np.sqrt(-hq.nsmallest(1, self._queue)[0].key)
