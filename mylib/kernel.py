import numpy as np


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
    return (
        (
            1 / (max_distance_h * np.sqrt(np.pi))
        ) * np.e**(-(current_radius**2 / max_distance_h**2))
    )


def monoghan_kernel(current_radius, max_distance_h) -> int:
    """
    param: current_radius: radius the currently calculated particle
    param: max_distance_h: distance to farthest particle
    """
    R_div_H = current_radius / max_distance_h  # pre-divide to improve performance
    NORM = 40 / (7 * np.pi)
    PREFACTOR = NORM / max_distance_h**2

    if current_radius >= 0 and R_div_H < 0.5:
        return PREFACTOR * (6 * (R_div_H ** 3) - 6 * (R_div_H**2) + 1)
    elif 0.5 <= R_div_H and R_div_H <= 1:
        return PREFACTOR * (2 * (1 - R_div_H)**3)
    elif R_div_H > 1:
        return 0
