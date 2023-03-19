import sys
sys.path.append('..')

from mylib.sph import SPH
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


if __name__ == "__main__":

    sph_run = SPH(500, 0.1)
    
    fig, axs = plt.subplots()
    part_pos = [sph_run.PARTICLES[i].r for i in range(
        len(sph_run.PARTICLES))]
    dim_0 = [sph_run.PARTICLES[i].r[0]
             for i in range(len(sph_run.PARTICLES))]
    dim_1 = [sph_run.PARTICLES[i].r[1]
             for i in range(len(sph_run.PARTICLES))]
    scatter = axs.scatter(dim_0, dim_1, c='r', marker='o') # type: ignore

    # print('before firs step: particle 0: ', sph_run.PARTICLES[0].r, sph_run.PARTICLES[0].velocity, sph_run.PARTICLES[0].accel)
    sph_run.first_step()
    # print('after firs step: particle 0: ', sph_run.PARTICLES[0].r, sph_run.PARTICLES[0].velocity, sph_run.PARTICLES[0].accel)

    def update(frame):
        global sph_run, scatter
        sph_run.update()
        scatter.set_offsets([sph_run.PARTICLES[i].r for i in range(len(sph_run.PARTICLES))])
        print('frame: ', frame)
        # print('particle 0: ', sph_run.PARTICLES[0].r)

    ani = FuncAnimation(fig, update, frames=range(100), interval=100, repeat=False)

    plt.show()
