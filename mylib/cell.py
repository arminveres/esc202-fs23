import numpy as np
from numpy.typing import NDArray

# from particle import Particle
from mylib.particle import Particle


class Cell:
    """
    Cell object with containes other cells as tree type
    """

    def __init__(
        self,
        regionLowerBound: NDArray[np.float64],
        regionHigherBound: NDArray[np.float64],
        lower_index: int,
        upper_index: int,
    ):
        self.regionLowerBound = regionLowerBound  # [xMin, yMin]
        self.regionHigherBound = regionHigherBound  # [xMax, yMax]
        self.iLower = lower_index  # index to first particle in particle array
        self.iUpper = upper_index  # index to last particle in particle array
        self.lowerCell = None  # reference to tree cell for lower part
        self.upperCell = None  # reference to tree cell for upper part
        self.rc = self._center()

    def celldist2(self, r: NDArray[np.float64]):
        """
        Calculates the squared minimum distance between a particle
        position and this node(cell).
        param: r particle position
        """
        d1 = r - self.regionHigherBound
        d2 = self.regionLowerBound - r
        d1 = np.maximum(d1, d2)
        d1 = np.maximum(d1, np.zeros_like(d1))
        return d1.dot(d1)

    def particle_is_inside(self, particle: Particle) -> bool:
        return (
            (particle.r[0] < self.regionHigherBound[0]) and
            (particle.r[0] > self.regionLowerBound[0]) and
            (particle.r[1] < self.regionHigherBound[1]) and
            (particle.r[1] > self.regionLowerBound[1])
        )

    def _center(self) -> NDArray[np.float64]:
        """
        Return the centre of the cell
        """
        return np.array([
            (self.regionLowerBound[0] + self.regionHigherBound[0]) / 2,
            (self.regionLowerBound[1] + self.regionHigherBound[1]) / 2,
        ])

    def __repr__(self):
        return f" {self.regionLowerBound}, {self.regionHigherBound}, {self.iLower}, {self.iUpper}\n"
