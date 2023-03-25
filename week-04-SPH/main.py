import sys
sys.path.append('..')

from mylib.sph import SPH
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("error")

if __name__ == "__main__":

    NO_PARTICLES = 100
    # smallest radius divided by c divide by 3
    sph_run = SPH(NO_PARTICLES, 0.00005)

    fig, axs = plt.subplots()
    dim_0 = [sph_run.PARTICLES[i].r[0]
             for i in range(len(sph_run.PARTICLES))]
    dim_1 = [sph_run.PARTICLES[i].r[1]
             for i in range(len(sph_run.PARTICLES))]
    dens = [sph_run.PARTICLES[i].rho
            for i in range(len(sph_run.PARTICLES))]
    scatter = axs.scatter(dim_0, dim_1, c='r', marker='o')  # type: ignore

    sph_run.first_step()

    def update(frame):
        global sph_run, scatter
        sph_run.update()
        scatter.set_offsets([sph_run.PARTICLES[i].r for i in range(len(sph_run.PARTICLES))])
        print('frame: ', frame)
        print('delta: ', sph_run.dt)

    ani = FuncAnimation(fig, update, frames=range(100), interval=100, repeat=False)
    ani.save(f'sph_{NO_PARTICLES}.mp4')
    # plt.show()
