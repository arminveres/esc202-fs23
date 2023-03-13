import numpy as np
from .particle import Particle


class SPH:
    def __init__(self, no_particles, timestep):
        self.no_particles = no_particles
        self.dt = timestep

        # generate partciles
        self.particles = [Particle(np.random.rand(2)) for _ in range(self.no_particles)]

    def __drift_one(self, delta=0):
        for particle in self.particles:
            particle.r += particle.velocity * delta
            particle.velocity_pred = particle.velocity + particle.accel * delta
            particle.energy_pred = particle.energy + particle.energy_dot * delta

    def __drift_two(self, delta=0):
        for particle in self.particles:
            particle.r += particle.velocity * delta

    def __kick(self, delta=0):
        for particle in self.particles:
            particle.velocity += particle.accel * delta
            particle.energy += particle.energy_dot * delta

    def __calculate_forces(self):
        # Treebuild
        # nn-density
        # calcsound
        # nn-sphforce
        pass

    def run(self, nsteps: int):
        self.__drift_one()
        self.__calculate_forces()
        for _ in range(nsteps):
            self.__drift_one(self.dt / 2)
            self.__calculate_forces()
            self.__kick(self.dt / 2)
            self.__drift_two(self.dt / 2)

    # in case we want to animate

    def first_step(self):
        self.__drift_one()
        self.__calculate_forces()

    def update(self):
        self.__drift_one(self.dt / 2)
        self.__calculate_forces()
        self.__kick(self.dt / 2)
        self.__drift_two(self.dt / 2)


