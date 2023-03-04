import numpy as np


class Cell:
    """
    Cell object with containes other cells as tree type
    """

    # TODO: add self.rc, centerpoint
    def __init__(
        self,
        regionLowerBound: np.ndarray[int, int],
        regionHigherBound: np.ndarray[int, int],
        lower_index: int,
        upper_index: int,
    ):
        self.regionLowerBound = regionLowerBound  # [xMin, yMin]
        self.regionHigherBound = regionHigherBound  # [xMax, yMax]
        self.iLower = lower_index  # index to first particle in particle array
        self.iUpper = upper_index  # index to last particle in particle array
        self.lowerCell = None  # reference to tree cell for lower part
        self.upperCell = None  # reference to tree cell for upper part
        self.rc = self.center()
        # self.is_leaf = False

    def celldist2(self, r: np.ndarray[int, int]):
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

    def dist_center_to_other(self, other: np.ndarray[int, int]):
        distance_squared = 0
        b_coord = [
            self.regionHigherBound[0] - self.rc[0],
            self.regionHigherBound[1] - self.rc[1],
        ]
        l_coord = [
            self.regionHigherBound[0] - self.regionLowerBound[0],
            self.regionHigherBound[1] - self.regionLowerBound[1],
        ]
        for d in range(2):
            tmp = np.abs(self.rc[d] - other[d])
            if tmp < 0:
                t1 = tmp + l_coord[d]
            else:
                t1 = tmp - l_coord[d]
            tmp = np.abs(tmp) - b_coord[d]
            t1 = abs(t1) - b_coord[d]
            if t1 < tmp:
                tmp = t1
            if tmp > 0:
                distance_squared += tmp**2
        return distance_squared

    def center(self) -> np.ndarray[int, int]:
        """
        Return the centre of the cell
        """
        # return (self.regionLowerBound / 2, self.regionHigherBound / 2)
        return [
            (self.regionLowerBound[0] + self.regionHigherBound[0]) / 2,
            (self.regionLowerBound[1] + self.regionHigherBound[1]) / 2,
        ]
