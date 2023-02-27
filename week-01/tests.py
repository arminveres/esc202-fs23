from treebuild import (
    Particle,
    partition,
)

import numpy as np

###################################################################################################
# Tests
###################################################################################################


def test_noParticles() -> bool:
    print("Running: test_noParticles")
    # A = initialize with particle with sequential coordinates in x, same yMax
    A = np.array([])
    s = partition(A, 0, 0, 0.5, 0)
    return s == None 


def test_allParticlesOnOneSide() -> bool:
    print("Running: test_allParticlesOnOneSide")
    A = np.array([Particle(np.array([1.0, 1.0]))])
    s = partition(A, 0, 0, 0.5, 0)
    return s == 0


def test_invertedOrder() -> bool:
    print("Running: test_invertedOrder")
    A = np.array(
        [
            Particle(np.array([1.0, 1.0])),
            Particle(np.array([0.9, 1.0])),
            Particle(np.array([0.8, 1.0])),
            Particle(np.array([0.3, 1.0])),
            Particle(np.array([0.0, 1.0])),
        ]
    )
    s = partition(A, 0, 5, 0.5, 0)
    return s == 1


def run_all_tests() -> bool:
    print("Running Tests ...\n")

    if not test_noParticles():
        print("test_noParticles failed.")
        return False

    if not test_allParticlesOnOneSide():
        print("test_allParticlesOnOneSide failed.")
        return False

    if not test_invertedOrder():
        print("test_invertedOrder failed.")
        return False

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    run_all_tests()
