import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import numpy.random as rd
from numba import jit

@jit(nopython=True)
def metropolis(s, beta, NX, NY):
    i = rd.randint(0, NX)
    j = rd.randint(0, NY)
    dE = deltae(s, i, j, NX, NY)
    if dE < 0:
        s[i, j] *= -1
    else:
        r = rd.random()
        if r < np.exp(-beta * dE):
            s[i, j] *= -1


@jit(nopython=True)
def deltae(s, i, j, NX, NY):
    summe = s[i, (j - 1) % NY] + s[i, (j + 1) % NY]
    summe += s[(i - 1) % NX, j] + s[(i + 1) % NX, j]
    return 2 * J * s[i, j] * summe


def update(frame):
    ln.set_data(s_list[frame])
    return (ln,)


def init():
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    return (ln,)


if __name__ == "__main__":
    NX = 150
    NY = 150
    s = np.ones((NX, NY))
    x_cntr = 0
    for x in s:
        y_cntr = 0
        for y in x:
            sign = rd.choice([-1, 1])
            s[x_cntr, y_cntr] *= sign
            y_cntr += 1
        x_cntr += 1

    J = 1
    steps = 40
    N_PER_TEMP = steps * NX * NY
    TEMP_START = 4
    TEMP_END = 0.1
    TEMP_FACTOR = 0.98
    temp = TEMP_START
    s_list = []
    cntr = 0
    while temp >= 0.1:
        temp *= 0.98
        beta = 1 / temp
        print(f"step no.: {cntr}")
        cntr += 1
        for l in range(N_PER_TEMP):
            metropolis(s, beta, NX, NY)
        s_list.append(np.copy(s))

    frame = len(s_list)
    # s_list = np.array(s_list)
    # animation
    fig, ax = plt.subplots()
    ln = ax.imshow(s_list[0], animated=True)
    ani = FuncAnimation(fig, update, frames=range(frame), init_func=init, blit=True)

    ani.save("ising2d.mp4", dpi=250)
    # plt.show()
