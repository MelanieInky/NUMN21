import numpy as np
from optimization import OptimizationProblem
from optimizer import NewtonOptimizer
import matplotlib.pyplot as plt

def Rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def gradRosenbrock(x):
    fdx1 = 400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2
    fdx2 = 200 * (x[1] - x[0]**2)
    return np.array([fdx1, fdx2])


if __name__ == '__main__':
    problem = OptimizationProblem(Rosenbrock, gradf = gradRosenbrock)
    optimizer = NewtonOptimizer(problem,1e-9,"exact")
    optimizer.solve(np.array([0,-0.7]),15)
    print(f'optimizer.xhist: {optimizer.xhist}')

    x = np.linspace(-0.5, 2, 1000)
    y = np.linspace(-1.5, 4, 1000)
    X, Y = np.meshgrid(x, y)
    Z = Rosenbrock([X,Y])
    levels = np.array([0.3, 9, 30, 100, 700, 1500, 3020])
    plt.contour(X, Y, Z, levels=levels, colors = 'black')
    plt.plot(np.array(optimizer.xhist)[:,0], np.array(optimizer.xhist)[:,1],'--', color = 'black')
    plt.plot(np.array(optimizer.xhist)[:,0], np.array(optimizer.xhist)[:,1],'.')
    plt.show()
