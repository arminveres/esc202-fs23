import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class TravelingSalesman:
    def __init__(self, file):
        """
        Load a given .tsp file and run the TSP on it.
        """
        self.nodes = np.loadtxt(
            file, delimiter=" ", comments="EOF", skiprows=6, usecols=(1, 2), unpack=True
        )
        self.N = len(self.nodes[0])
        self.nodes = self.nodes.reshape((self.N, 2)) * 0.001

        self.path = np.arange(self.N)

        # initialize T0
        max_energy = 0
        for _ in range(100):
            e, p = self.try_swap()
            max_energy = max(e, max_energy)
        self.T = max_energy

        # initialize path
        min_energy = self.N * 100
        min_path = self.path
        for _ in range(1000):
            e, p = self.try_path()
            if e < min_energy:
                min_energy = e
                min_path = p
        self.path = min_path

    def do_step(self):
        """
        Do a single step
        """
        for _ in range(500):
            e_prime, p = self.try_swap()
            e = self.calculate_energy(self.path)
            if e_prime < e or rd.random() < np.exp((e - e_prime) / self.T):
                self.path = p

            e_prime, p = self.try_flip()
            e = self.calculate_energy(self.path)
            if e_prime < e or rd.random() < np.exp((e - e_prime) / self.T):
                self.path = p

        self.T *= 0.9

    def get_map(self):
        """
        Return the current map index
        """
        return np.append(self.path, self.path[0])

    def calculate_energy(self, path):
        result = np.roll(self.nodes[path], 1, axis=0)
        result -= self.nodes[path]
        result = np.multiply(result, result)
        result = np.sum(result, 1)
        result = np.sqrt(result)
        return sum(result)

    def try_path(self):
        linear = np.arange(self.N)
        rd.shuffle(linear)
        return self.calculate_energy(linear), linear

    def try_swap(self):
        """
        swap two random nodes on virtual path and return new path
        """
        idx = rd.randint(0, high=self.N, size=2)
        path_copy = np.copy(self.path)
        self.path[idx[0]], self.path[idx[1]] = self.path[idx[1]], self.path[idx[0]]
        return self.calculate_energy(path_copy), path_copy

    def try_flip(self):
        idx = rd.randint(0, high=self.N, size=2)
        path_copy = np.copy(self.path)
        path_copy[idx[0] : idx[1]] = np.flip(path_copy[idx[0] : idx[1]])
        return self.calculate_energy(path_copy), path_copy


if __name__ == "__main__":
    tsp = TravelingSalesman("./ch130.tsp")
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    axLeft = fig.add_subplot(gs[:, 0])
    axRight = fig.add_subplot(gs[:, 1])

    (pathPlot,) = axLeft.plot(
        tsp.nodes[tsp.get_map(), 0],
        tsp.nodes[tsp.get_map(), 1],
    )
    nodePlot = axLeft.scatter(
        tsp.nodes[tsp.get_map(), 0],
        tsp.nodes[tsp.get_map(), 1],
        c="black",
    )
    (ePlot,) = axRight.plot([], [])
    (tPlot,) = axRight.plot([], [])
    axRight.set_xlim(0, 1)
    axRight.set_ylim(0, tsp.N * 0.6)
    no_frames = 400
    times = []
    energy = []
    temps = []

    def update(time):
        global tsp
        tsp.do_step()
        times.append(time / no_frames)
        energy.append(tsp.calculate_energy(tsp.path))
        temps.append(tsp.T)
        pathPlot.set_data(
            tsp.nodes[tsp.get_map(), 0],
            tsp.nodes[tsp.get_map(), 1],
        )
        nodePlot.set_offsets(tsp.nodes[tsp.get_map(), :])
        ePlot.set_data(times, energy)
        tPlot.set_data(times, temps)

    ani = FuncAnimation(fig, update, frames=range(no_frames), interval=10, repeat=False)
    ani.save(f"tsp_{no_frames}.mp4")
    plt.show()
