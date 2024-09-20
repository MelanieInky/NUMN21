from finite_diff import finite_difference
import numpy as np
from optimization import OptimizationProblem


class Optimizer:
  def __init__(self,problem):
    self.problem = problem
    pass

  def step(self,x):
    pass

  def solve(self,x0, max_iter = 20):
    x = x0
    self.xhist = [x]
    for i in range(max_iter):
      x, stop = self.step(x)
      self.xhist.append(x)
      if stop:
        self.success = True
        return x
    self.success = False
    print('Optimizer did not converge')
    return x
    pass

class NewtonOptimizer(Optimizer):
  def __init__(self,problem,stop_threshold = 1e-6):
    super().__init__(problem)
    self.stop_threshold = stop_threshold
    pass

  def step(self,x):
    hessian = finite_difference(problem.gradf,x,problem.epsilon,**problem.kwargs)
    grad = problem.gradf(x,**problem.kwargs)
    dx = np.linalg.solve(hessian,grad)
    x = x - dx
    stop = np.linalg.norm(dx) < self.stop_threshold
    return x, stop
    pass

class QuasiNewtonOptimizer(Optimizer):
  def __init__(self,problem):
    super().__init__(problem)
    pass

  def step(self,x0):
    pass


if __name__ == '__main__':
    
    def g(x,r):
        return np.sum(r*x**2)

    def grad_g(x,r):
        return r*2*x

    epsilon = 1e-6
    problem = OptimizationProblem(g, gradf = grad_g, r = 2)
    gradtest = problem.gradf(np.array([1,2]),**problem.kwargs)
    hesstest = finite_difference(problem.gradf,np.array([1,2]),epsilon,**problem.kwargs)
    print(f'hesstest: {hesstest}')
    print(f'gradtest: {gradtest}')

    optimizer = NewtonOptimizer(problem,1e-8)
    optimizer.solve(np.array([-8,6]),2)
    print(f'optimizer.xhist: {optimizer.xhist}')