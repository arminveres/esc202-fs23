import numpy as np


class Cell:
    """
    Cell object with containes other cells as tree type
    """

    # TODO: add self.rc, centerpoint
    def __init__(
        self,
        regionLowerBound: np.ndarray,
        regionHigherBound: np.ndarray,
        lower_index: int,
        upper_index: int,
    ):
        self.regionLowerBound = regionLowerBound  # [xMin, yMin]
        self.regionHigherBound = regionHigherBound  # [xMax, yMax]
        self.iLower = lower_index  # index to first particle in particle array
        self.iUpper = upper_index  # index to last particle in particle array
        self.lowerCell = None  # reference to tree cell for lower part
        self.upperCell = None  # reference to tree cell for upper part

    def celldist2(self, r):
        """
        Calculates the squared minimum distance between a particle
        position and this node(cell).
        """
        d1 = r - self.rHigh
        d2 = self.rLow - r
        d1 = np.maximum(d1, d2)
        d1 = np.maximum(d1, np.zeros_like(d1))
        return d1.dot(d1)
