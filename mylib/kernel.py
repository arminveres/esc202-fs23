import numpy as np
from numpy.typing import NDArray


NORM = 40 / (7 * np.pi)


def top_hat_kernel(current_radius, max_distance_h):
    """
    param: current_radius: radius the currently calculated particle
    param: max_distance_h: distance to farthest particle
    """
    return 1 / ((max_distance_h**2) * np.pi)


def gaussian_kernel(current_radius, max_distance_h):
    """
    param: current_radius: radius the currently calculated particle
    param: max_distance_h: distance to farthest particle
    """
    return (1 / (max_distance_h * np.sqrt(np.pi))) * np.e ** (
        -(current_radius**2 / max_distance_h**2)
    )


def monoghan_kernel(r_current_radius, h_max_distance) -> float:
    """
    param: current_radius: radius the currently calculated particle
    param: max_distance_h: distance to farthest particle
    """
    global NORM
    R_DIV_H = r_current_radius / h_max_distance  # pre-divide to improve performance
    PREFACTOR = NORM / h_max_distance**2

    if r_current_radius >= 0 and R_DIV_H < 0.5:
        return PREFACTOR * (6 * (R_DIV_H**3) - 6 * (R_DIV_H**2) + 1)
    elif 0.5 <= R_DIV_H and R_DIV_H <= 1:
        return PREFACTOR * (2 * (1 - R_DIV_H) ** 3)
    elif R_DIV_H > 1:
        return 0
    # raise ValueError
    return 0


def derivative_monoghan(r_current_radius: float, h_max_distance: float) -> float:
    global NORM
    PREFACTOR = (6 * NORM) / (h_max_distance**3)
    R_DIV_H = r_current_radius / h_max_distance  # pre-divide to improve performance
    if r_current_radius >= 0 and R_DIV_H < 0.5:
        return PREFACTOR * (3 * (R_DIV_H**2) - 2 * R_DIV_H)
    elif 0.5 <= R_DIV_H and R_DIV_H <= 1:
        return PREFACTOR * (-((1 - R_DIV_H) ** 2))
    return 0.0


def gradient_monoghan(
    r_a: NDArray[np.float64], r_b: NDArray[np.float64], max_dist_h: float
) -> float:
    pos_diff = r_a - r_b

    # get smallest distance
    box_size = np.array([1.0, 1.0])
    pos_diff -= np.round(pos_diff / box_size) * box_size
    abs_pos_diff = np.linalg.norm(pos_diff)

    # print(f"absposdiff: {abs_pos_diff}, max_dist_h: {max_dist_h}")
    derivative = derivative_monoghan(abs_pos_diff, max_dist_h)
    # print("derivative; ", derivative)
    gradient = (derivative * pos_diff) / abs_pos_diff
    # gradient = (derivative * r_a) / abs_pos_diff
    # print("gradient: ", gradient)
    return gradient


def wendtland_kernel(
    r_a: NDArray[np.float64], r_b: NDArray[np.float64], max_dist_h: float
) -> float:
    pos_diff = r_a - r_b
    h = max_dist_h / 2
    q = np.linalg.norm(pos_diff) / h
    if 0 <= q <= 2:
        return ((1 - q / 2) ** 4) * (1 + 2 * q)
    elif 2 < q:
        return 0


def gradient_wendtland():
    pass
