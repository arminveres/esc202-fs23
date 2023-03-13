import numpy as np


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
    return (
        (
            1 / (max_distance_h * np.sqrt(np.pi))
        ) * np.e**(-(current_radius**2 / max_distance_h**2))
    )


def monoghan_kernel(r_current_radius, h_max_distance) -> int:
    """
    param: current_radius: radius the currently calculated particle
    param: max_distance_h: distance to farthest particle
    """
    R_DIV_H = r_current_radius / h_max_distance  # pre-divide to improve performance
    PREFACTOR = NORM / h_max_distance**2

    if r_current_radius >= 0 and R_DIV_H < 0.5:
        return PREFACTOR * (6 * (R_DIV_H ** 3) - 6 * (R_DIV_H**2) + 1)
    elif 0.5 <= R_DIV_H and R_DIV_H <= 1:
        return PREFACTOR * (2 * (1 - R_DIV_H)**3)
    elif R_DIV_H > 1:
        return 0


def derivative_monoghan(r_current_radius, h_max_distance):
    PREFACTOR = (6 * NORM) / (h_max_distance**2)
    R_DIV_H = r_current_radius / h_max_distance  # pre-divide to improve performance
    if r_current_radius >= 0 and R_DIV_H < 0.5:
        PREFACTOR * (3 * (R_DIV_H**2) - 2 * R_DIV_H)
    elif 0.5 <= R_DIV_H and R_DIV_H <= 1:
        PREFACTOR * (-(1 - (R_DIV_H**2)))


def gradient_monoghan(radius_a, radius_b, max_dist_h):
    radius_ab = radius_a - radius_b
    
