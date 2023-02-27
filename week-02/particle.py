import numpy as np


class Particle:
    """
    Particle class with corresponding properties.
    """
    def __init__(self, r: np.ndarray):
        self.r: np.ndarray = r  # positionn of the particle [x, y]
        self.rho = 0.0  # density of the particle
        # ...  more properties of the particle
