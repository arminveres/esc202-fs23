import numpy as np
from mylib.cell import Cell
from mylib.neigbour_search import neighbor_search_periodic
from mylib.kernel import gradient_monoghan, monoghan_kernel, top_hat_kernel
from mylib.treebuild import build_tree
from mylib.particle import Particle, PriorityQueue


class SPH:
    def __init__(self, no_particles, timestep):
        self.no_particles = no_particles
        self.dt = timestep
        self.gamma = 2
        # generate particles
        self.PARTICLES: list[Particle] = [Particle(np.random.rand(2))
                          for _ in range(self.no_particles)]

    def __drift_one(self, delta=0):
        for particle in self.PARTICLES:
            particle.r += particle.velocity * delta
            particle.velocity_pred = particle.velocity + particle.accel * delta
            particle.energy_pred = particle.energy + particle.energy_dot * delta

    def __drift_two(self, delta=0):
        for particle in self.PARTICLES:
            particle.r += particle.velocity * delta
            # maybe we should wrap around our area and take the absolute value!
            particle.r[0] = np.abs(particle.r[0] % 1.0)
            particle.r[1] = np.abs(particle.r[1] % 1.0)

    def __kick(self, delta=0):
        for particle in self.PARTICLES:
            particle.velocity += particle.accel * delta
            particle.energy += particle.energy_dot * delta

    def __calculate_forces(self):
        # TODO: (avee) parallelize this part !!!

        # Treebuild
        root = Cell(
            regionLowerBound=np.array([0.0, 0.0]),
            regionHigherBound=np.array([1.0, 1.0]),
            lower_index=0,
            upper_index=len(self.PARTICLES) - 1,
        )

        build_tree(self.PARTICLES, root, 0)

        K = 32  # number of neighbours
        
        # nn-density
        for particle in self.PARTICLES:
            particle.priority_queue = PriorityQueue(K)
            sum_of_mass = 0

            neighbor_search_periodic(particle.priority_queue, root, self.PARTICLES, particle.r, np.array([1, 1]))

            H = np.sqrt(particle.priority_queue.key())
            for i in range(K):
                R = np.sqrt(-particle.priority_queue._queue[i].key)
                # get the mass of each neighbours
                mass = particle.priority_queue._queue[i].mass
                rho = mass * monoghan_kernel(R, H)
                sum_of_mass += rho

            # FIXME: (avee) This may result in 0, which in turn generates a division by zero error at line 83
            particle.rho = sum_of_mass
            # print(particle.rho)
            # calcsound
            # particle.c_speed_sound = np.sqrt(np.abs(particle.energy) * self.gamma * (self.gamma - 1))
            particle.c_speed_sound = np.sqrt(particle.energy * self.gamma * (self.gamma - 1))

        # nn-sphforce
        for particle in self.PARTICLES:
            # particle.energy_dot = 0.0
            f_a = (particle.c_speed_sound ** 2) / (2.0 * particle.rho)
            h_max = particle.priority_queue.get_max_distance() # current max distance
            for near_particle in particle.priority_queue._queue:  # foreach neighbour
                if (particle.r != near_particle.r).all():  # skip if the same particle
                    f_b = (near_particle.c_speed_sound ** 2) / (2.0 * particle.rho)
                    particle.energy_dot += near_particle.mass * (particle.velocity - near_particle.velocity).dot(gradient_monoghan(particle.r, near_particle.r, h_max))

                    # FIXME: particles disappear and likely cause is energy_dot being negative
                    # particle.energy_dot = np.abs(particle.energy_dot)
                    print(particle.energy_dot)
                    particle.accel -= near_particle.mass * (f_a+f_b)*gradient_monoghan(particle.r, near_particle.r, h_max)

            particle.energy_dot *= f_a

    def run(self, nsteps: int):
        self.__drift_one()
        self.__calculate_forces()
        for _ in range(nsteps):
            self.__drift_one(self.dt / 2)
            self.__calculate_forces()
            self.__kick(self.dt)
            self.__drift_two(self.dt / 2)
            print(
"""
=======================
Step done
=======================
"""
                )
        

    # in case we want to animate

    def first_step(self):
        self.__drift_one()
        self.__calculate_forces()

    def update(self):
        self.__drift_one(self.dt / 2)
        # print('drift one: ', self.PARTICLES[0].r, self.PARTICLES[0].velocity, self.PARTICLES[0].accel)
        self.__calculate_forces()
        # print('calc forces: ', self.PARTICLES[0].r , self.PARTICLES[0].velocity , self.PARTICLES[0].accel)
        self.__kick(self.dt)
        # print('kick: ', self.PARTICLES[0].r, self.PARTICLES[0].velocity, self.PARTICLES[0].accel)
        self.__drift_two(self.dt / 2)
        # print('drift two: ', self.PARTICLES[0].r, self.PARTICLES[0].velocity, self.PARTICLES[0].accel)