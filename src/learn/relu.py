import matplotlib.pyplot as plt
import numpy as np

def plot_relu():
    xs = np.arange(-4.0, 4.1, 0.25)

    def relu(x):
        if x <= 0:
            return 0
        else:
            return x
        
    ys = list(map(lambda x: relu(x), xs))
    print(ys)
    plt.plot(xs, ys)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()