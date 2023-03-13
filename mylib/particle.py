import numpy as np


class Particle:
    """
    Particle class with corresponding properties.
    """

    # TODO: (aver) add the prio queues

    def __init__(self, r: np.ndarray[int, int], key=-np.inf):
        self.r = r  # position of the particle [x, y]
        self.accel = 0.0
        self.velocity = 0.0
        self.velocity_pred = 0.0
        self.rho = 0.0  # density of the particle
        self.mass = 1
        self.energy = 10
        self.energy_pred = 0.0
        self.energy_dot = 0.0
        # c: speed of sound
        self.c_speed_sound = 0.0

        # h: distance to neighbout
        self.key = key  # meant as an implemetation for the priority queue

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
        return f"Pos: {self.r}, Dens: {self.rho}, PQ Key: {self.key}\n"
