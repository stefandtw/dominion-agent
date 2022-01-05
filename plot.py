import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title(f'Running average of previous 100 scores ({figure_file})')
    plt.savefig(figure_file)
    plt.show()
