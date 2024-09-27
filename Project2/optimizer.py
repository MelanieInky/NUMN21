from finite_diff import finite_difference
import numpy as np
from optimization import OptimizationProblem
from abc import ABC, abstractmethod

class Optimizer(ABC):
  def __init__(self,problem,stop_threshold = 1e-6):
    self.problem = problem
    self.stop_threshold = 1e-6
    pass


  def step(self,x):
    s = self.calculate_s()
    alpha = 1 #To be added, find with line search
    x = x + alpha * s
    stop = np.linalg.norm(s) < self.stop_threshold
    return x, stop
  
  @abstractmethod
  def calculate_s(self):
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
    super().__init__(problem,stop_threshold)
    #self.stop_threshold = stop_threshold
    pass

  def calculate_s(self):
    x = self.xhist[-1] 
    hessian = finite_difference(problem.gradf,x,problem.epsilon,**problem.kwargs)
    grad = problem.gradf(x,**problem.kwargs)
    s = -np.linalg.solve(hessian,grad)
    return s

class QuasiNewtonOptimizer(Optimizer):
  def __init__(self,problem,stop_threshold = 1e-6):
    super().__init__(problem,stop_threshold)
    pass
  
  def calculate_s(self):
    try: #If H already exists
      H = self.H
      xnew = self.xhist[-1]
      x = self.xhist[-2]
      #TODO, don't compute the gradient twice, save it instead!
      gnew = self.problem.gradf(xnew,**self.problem.kwargs)
      g = self.problem.gradf(x,**self.problem.kwargs)
      self.H = self.calculate_H(H,gnew,g,xnew,x)
    except AttributeError: #In the case of the first step, do the usual stuff
      xnew = self.xhist[-1] 
      hessian = finite_difference(self.problem.gradf,xnew,problem.epsilon,**self.problem.kwargs)
      gnew = self.problem.gradf(xnew,**problem.kwargs)
      self.H = np.linalg.inv(hessian)
    s = - self.H@gnew
    return s
  
  @abstractmethod
  def calculate_H(self):
    pass
    



class GoodBroyden(QuasiNewtonOptimizer):
    def __init__(self,problem,stop_threshold = 1e-6):
      super().__init__(problem,stop_threshold)

    def calculate_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian 
        - g: gradient of objective funtion at the previous point x
        - gnew: gradient of objective funtion at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x    #displacement between the new point xnew and the old point x
        y = gnew - g    #difference between in gradients between the new point xnew and the old point x 
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d - H @ y) / (d.T @ H @ y) @ d.T @ H    #update Inverse Hessian by Sherman-Morisson formula
        return Hnew


class BadBroyden(QuasiNewtonOptimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective function at the previous point x
        - gnew: gradient of objective function at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x    #displacement between the new point xnew and the old point x
        y = gnew - g    #difference between in gradients between the new point xnew and the old point x
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d - H @ y) / (y.T @ y) @ y.T #update

        return Hnew
    
class SymmetricBroyden(QuasiNewtonOptimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective function at the previous point x
        - gnew: gradient of objective function at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x    #displacement between the new point xnew and the old point x
        y = gnew - g    #difference between in gradients between the new point xnew and the old point x

        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        u = d - H @ y   #difference between displacement d and the predicted change based on the current Hessian approximation
        a = 1 / (u.T @ y)   #the reciprocal the dot product of the vector u with the gradient change y

        Hnew = H + a * u @ u.T  #update

        return Hnew

class DFP(QuasiNewtonOptimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective funtion at the previous point x
        - gnew: gradient of objective funtion at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x    #displacement between the new point xnew and the old point x
        y = gnew - g    #difference between in gradients between the new point xnew and the old point x
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d @ d.T) / (d.T @ y) - (H @ y @ y.T @ H) / (y.T @ H @ y)    #update
        return Hnew        
    
class BFGS(QuasiNewtonOptimizer):
    def calculate_H(self, H, gnew, g, xnew, x):
        d = xnew - x    #displacement between the new point xnew and the old point x
        y = gnew - g    #difference between in gradients between the new point xnew and the old point x
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        #dTy = d.T @ y
        #dyT = d @ y.T

        Hnew = (
            H
            + (1 + (y.T @ H @ y) / d.T @ y) * (d @ d.T) / d.T @ y
            - (d @ y.T @ H + H @ y @ d.T) / (d.T @ y)
        )

        return Hnew

    def optimize(self, x0):
        x_list = [x0]
        n = x0.shape[0]
        xnew = x0
        gnew = self.problem.gradient_function(x0)
        Hnew = np.eye(n)
        # self.points.append(copy.deepcopy(xnew))

        for _ in range(self.max_iterations):
            x = xnew
            g = gnew
            H = Hnew
            # print(H)

            s = -Hnew @ gnew
            # alpha, *_ = self.line_search.search(x, s, 0, 1e8)
            alpha, *_ = self.line_search.search(x, s)

            xnew = x + alpha * s
            gnew = self.problem.gradient_function(xnew)
            Hnew = self.calculate_H(H, gnew, g, xnew, x)
            x_list.append(xnew)

            # self.points.append(copy.deepcopy(xnew))
            if self.check_criterion(x, xnew, g):
                self.success = True
                self.xmin = xnew
                break

        return x_list  
      
      
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

    optimizer = GoodBroyden(problem,1e-8)
    optimizer.solve(np.array([-8,6]),2)
    print(f'optimizer.xhist: {optimizer.xhist}')

    #blabkla