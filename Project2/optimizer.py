from finite_diff import finite_difference
import numpy as np
from optimization import OptimizationProblem
from scipy.optimize import fsolve
from abc import ABC, abstractmethod

class Optimizer(ABC):
  def __init__(self,problem):
    self.problem = problem
    pass

  def exact_line_search(x, s):
    def dphi(alpha):
      return problem.gradf(x - alpha * s, **problem.kwargs)
    alpha_new = fsolve(dphi, [1, 1])
    return alpha_new

  def inexact_line_search(x, s):
    sigma = 0.9 #0.9 for weak, 0.1 for fairly accurate line search?
    rho = 0.01  #p30
    tau2 = 1/10 #p36
    tau3 = 1/2  #p36
    f_ = 0      #lower bound f(alpha)?
    f0 = problem.f(x - s, **problem.kwargs)
    df0 = problem.gradf(x - s, **problem.kwargs)
    mu = (f_ - f0) / (rho * df0)
    alpha = mu  #0 < a_1 <= mu?
    alpha_old = 0
    B = 0
    for i in range(10):
      alpha_old = alpha
      alpha = alpha_new
      fa = problem.f(x - alpha * s, **problem.kwargs)
      if fa <= f_:
        break
      fa_old = problem.f(x - alpha_old * s, **problem.kwargs)
      dfa = problem.gradf(x - alpha * s, **problem.kwargs)
      if fa > f0 + alpha * df0 or fa >= fa_old:
        a = alpha_old
        b = alpha
        B = 1
        break
      if abs(dfa) <= - sigma * df0:
        break
      if dfa >= 0:
        a = alpha
        b = alpha_old
        B = 1
        break
      if mu <= 2 * alpha - alpha_old:
        alpha_new = mu
      else:
        alpha_new = 2 * alpha - alpha_old # in [2 * alpha - alpha_old, min(mu, alpha + tau1 * (alpha - alpha_old))?
    if B == 1:
      for i in range(10):
        alpha = a + tau2 * (b - a) # in [a + tau2 * (b - a), b - tau3 * (b - a)]
        fa = problem.f(x - alpha * s, **problem.kwargs)
        f_a = problem.f(x - a * s, **problem.kwargs)
        if fa > f0 + rho * alpha * df0 or fa >= f_a:
          a_new = a
          b_new = alpha
          break
        else:
          dfa = problem.gradf(x - alpha * s, **problem.kwargs)
          if abs(dfa) <= - sigma * df0:
            break
          a_new = alpha
          if (b - a) * dfa >= 0:
            b_new = a
          else:
            b_new = b
    return alpha
    
  

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
    super().__init__(problem)
    self.stop_threshold = stop_threshold
    pass

  def calculate_s(self):
    x = self.xhist[-1] 
    hessian = finite_difference(problem.gradf,x,problem.epsilon,**problem.kwargs)
    grad = problem.gradf(x,**problem.kwargs)
    s = -np.linalg.solve(hessian,grad)
    return s

class QuasiNewtonOptimizer(Optimizer):
  def __init__(self,problem):
    super().__init__(problem)
    pass



class GoodBroyden(Optimizer):
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


class BadBroyden(Optimizer):
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

        Hnew = H + (d - H @ y) / (y.T @ y) @ y.T #update

        return Hnew
    
class SymmetricBroyden(Optimizer):
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

        u = d - H @ y   #difference between displacement d and the predicted change based on the current Hessian approximation
        a = 1 / (u.T @ y)   #the reciprocal the dot product of the vector u with the gradient change y

        Hnew = H + a * u @ u.T  #update

        return Hnew

class DFP(Optimizer):
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
    
class BFGS(Optimizer):
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

    optimizer = NewtonOptimizer(problem,1e-8)
    optimizer.solve(np.array([-8,6]),2)
    print(f'optimizer.xhist: {optimizer.xhist}')

    #blabkla