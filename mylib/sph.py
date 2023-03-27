import numpy as np
from .cell import Cell
from .neigbour_search import neighbor_search_periodic
from .kernel import gradient_monoghan, monoghan_kernel
from .treebuild import build_tree
from .particle import Particle, PriorityQueue

frame_counter = 0


class SPH:
    def __init__(self, no_particles, timestep, calc_dt=False):
        self.no_particles = no_particles
        self.dt = timestep
        self.gamma = 2
        self.__smallest_radius = np.inf
        self.__largest_soundspeed = -np.inf
        self.__calc_dt=calc_dt
        # generate particles
        self.PARTICLES: list[Particle] = [Particle(np.random.rand(2))
                                          for _ in range(self.no_particles)]
        # self.PARTICLES[4].energy = 1000.0

    def pi_ab(self, part_a: Particle, part_b: Particle) -> float:
        """
        Calculates artificial viscosity
        """
        ETA = 1e-3
        ALPHA = 1.0
        BETA = 2 * ALPHA
        r_ab = part_a.r.dot(part_b.r)
        v_ab = part_a.velocity.dot(part_b.velocity)

        q = v_ab * r_ab
        if q < 0:
            c_ab = 0.5 * (part_a.c_speed_sound + part_b.c_speed_sound)
            h_ab = 0.5 * (part_a.priority_queue.key() + part_b.priority_queue.key())
            rho_ab = 0.5 * (part_a.rho + part_b.rho)
            mu_ab = (h_ab * v_ab * r_ab) / (r_ab**2 + ETA**2)
            return (- ALPHA * c_ab * mu_ab + BETA * mu_ab ** 2) / rho_ab

        elif q >= 0:
            return 0.0

    def __drift_one(self, delta=0):
        for particle in self.PARTICLES:
            particle.r += particle.velocity * delta
            # we should wrap around our area and take the absolute value!
            particle.r = particle.r % 1.0

            particle.velocity_pred = particle.velocity + particle.accel * delta
            particle.energy_pred = particle.energy + particle.energy_dot * delta

    def __drift_two(self, delta=0):
        for particle in self.PARTICLES:
            particle.r += particle.velocity * delta
            # we should wrap around our area and take the absolute value!
            particle.r = particle.r % 1.0

    def __kick(self, delta=0):
        for particle in self.PARTICLES:
            particle.velocity += particle.accel * delta
            particle.energy += particle.energy_dot * delta
            if particle.energy < 0.:
                print("particle energy therefore dot < 0",particle.energy, particle.velocity, particle.energy_dot)

    def __nn_density(self):
        ###########################################################################
        # Treebuild
        ###########################################################################
        root = Cell(
            regionLowerBound=np.array([0.0, 0.0]),
            regionHigherBound=np.array([1.0, 1.0]),
            lower_index=0,
            upper_index=len(self.PARTICLES) - 1,
        )
        build_tree(self.PARTICLES, root, 0)

        ###########################################################################
        # Density
        ###########################################################################
        K = 32  # number of neighbours
        for particle in self.PARTICLES:
            particle.priority_queue = PriorityQueue(K)
            rho = 0

            neighbor_search_periodic(particle.priority_queue, root,
                                     self.PARTICLES, particle.r, np.array([1, 1]))

            H = np.sqrt(particle.priority_queue.key())
            for i in range(K):
                R = np.sqrt(-particle.priority_queue._queue[i].key)
                mass = particle.priority_queue._queue[i].mass
                rho += mass * monoghan_kernel(R, H)
            particle.rho = rho

            # calcsound
            try:
                particle.c_speed_sound = np.sqrt(particle.energy * self.gamma * (self.gamma - 1))
            # particle.c_speed_sound = np.sqrt(particle.energy_pred * self.gamma * (self.gamma - 1))
            except RuntimeWarning:
                particle.c_speed_sound = np.sqrt(
                    np.abs(particle.energy) * self.gamma * (self.gamma - 1))
                print(
                    f"Negative Energy: {particle.energy}, {particle.energy_pred}, {particle.c_speed_sound} on particle")

            if self.__calc_dt:
                self.__largest_soundspeed = particle.c_speed_sound \
                    if particle.c_speed_sound > self.__largest_soundspeed \
                    else self.__largest_soundspeed

    def __nn_sphforces(self):
        for particle in self.PARTICLES:
            # reset particle properties
            particle.energy_dot = 0
            particle.accel = 0

            if self.__calc_dt:
                self.__smallest_radius = particle.priority_queue.get_max_distance() \
                    if self.__smallest_radius > particle.priority_queue.get_max_distance() \
                    else self.__smallest_radius

            # BUG: How to fix fa/fb being infinite?
            f_a = (particle.c_speed_sound ** 2) / (self.gamma * particle.rho)
            h_max = particle.priority_queue.get_max_distance()  # current max distance
            if h_max <= 0:
                print("H max smaller equal than 0", h_max)

            for near_particle in particle.priority_queue._queue:  # foreach neighbour
                if (particle.r == near_particle.r).all():  # skip if the same particle
                    continue

                f_b = (near_particle.c_speed_sound ** 2) / (self.gamma * near_particle.rho)

                # calculate energy_dot
                particle.energy_dot += \
                    near_particle.mass * (particle.velocity - near_particle.velocity).dot(
                        gradient_monoghan(particle.r, near_particle.r, h_max))

                # FIXME: particles disappear and likely cause is energy_dot being negative
                if np.isnan(particle.energy_dot.all()):
                    print('nan found:', particle.energy_dot)

                # add acceleration
                visc = self.pi_ab(particle, near_particle)
                particle.accel +=  near_particle.mass * (
                        f_a + f_b + visc
                    ) * gradient_monoghan(particle.r, near_particle.r, h_max)

            particle.accel *= -1
            particle.energy_dot *= f_a

        ###########################################################################

    def __calculate_forces(self):
        self.__nn_density()
        self.__nn_sphforces()
        if self.__calc_dt:
            self.dt = self.__smallest_radius / self.__largest_soundspeed / 3

    def run(self, nsteps: int):
        self.__drift_one()
        self.__calculate_forces()
        for _ in range(nsteps):
            self.__drift_one(self.dt / 2)
            self.__calculate_forces()
            self.__kick(self.dt)
            self.__drift_two(self.dt / 2)
            print("""====================\nStep done\n====================""")

    # in case we want to animate, run first_step() first and then call update() subsequently

    def first_step(self):
        self.__drift_one()
        self.__calculate_forces()

    def update(self):
        global frame_counter
        self.__drift_one(self.dt / 2)
        self.__calculate_forces()
        self.__kick(self.dt)
        self.__drift_two(self.dt / 2)
        frame_counter += 1
