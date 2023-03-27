import numpy as np
import heapq as hq

from numpy.typing import NDArray


class Particle:
    """
    Particle class with corresponding properties.
    """

    def __init__(self, r: NDArray[np.float64], key=-np.inf):
        self.r = r  # position of the particle [x, y]
        self.accel = np.zeros((2))
        self.velocity = np.zeros((2))
        self.velocity_pred = np.zeros((2))
        self.rho = 0.0  # density of the particle
        self.mass = 1.0
        self.energy = 10.0  # np.random.random((1))
        self.energy_pred = np.zeros((1))
        self.energy_dot = np.zeros((1))
        # c: speed of sound
        self.c_speed_sound = np.zeros((1))
        # h: distance to neighbout
        self.key = key  # meant as an implemetation for the priority queue
        self.priority_queue: PriorityQueue = None

    def __eq__(self, other):  # for the == operator (equality)
        return self.key == other.key

    def __ne__(self, other):  # for the != operator (inequality)
        return self.key != other.key

    def __lt__(self, other):  # for the < operator (less than)
        return self.key < other.key

    def __le__(self, other):  # for the <= operator (less than or equal to)
        return self.key <= other.key

    def __gt__(self, other):  # for the > operator (greater than)
        return self.key > other.key

    def __ge__(self, other):  # for the >= operator (greater than or equal to)
        return self.key >= other.key

    def __repr__(self):
        return f"Posisition: {self.r}\n\
Accel:          {self.accel}\n\
Velocity:       {self.velocity}\n\
Speed of Sound: {self.c_speed_sound}\n\
Dens:           {self.rho}\n\
PQ Key:         {self.key}\n"


class PriorityQueue:
    """
    Priority Queue using the heapq algorithm
    """

    def __init__(self, queue_len_k: int):
        self._queue: list[Particle] = []
        # we instantiate the heap with infinite values
        for _ in range(queue_len_k):
            sentinel = Particle(np.array([np.float64(-np.inf), np.float64(-np.inf)]), -np.inf)
            hq.heappush(self._queue, sentinel)

    def __repr__(self):
        return str(self._queue)

    def key(self):
        """
        Return the key of highest key of an item in the priority queue
        """
        # need to reverse the minus sign, since the heapq returns the smallest item at 0 position
        return -self._queue[0].key

    def replace(self, particle: Particle) -> Particle:
        """
        Replaces the head(largest) element with given 'item', and returns it
        """
        return hq.heapreplace(self._queue, particle)

    def get_max_distance(self) -> float:
        # return np.sqrt(-hq.nsmallest(1, self._queue)[0].key)
        return np.sqrt(self.key())
