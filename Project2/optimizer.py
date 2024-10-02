# from Project2.finite_diff import finite_difference
from finite_diff import finite_difference
import numpy as np

# from Project2.optimization import OptimizationProblem
from optimization import OptimizationProblem
from scipy.optimize import fsolve
from abc import ABC, abstractmethod
import numpy.typing as npt


class Optimizer(ABC):
    def __init__(
        self, problem, stop_threshold=1e-6, line_search="none", rho=0.01, sigma=0.1
    ):
        """Generic optimizer class.

        Args:
            problem (OptimizationProblem): An optimization problem object
            stop_threshold (float, optional): The threshold at which the solver iterations stops. Defaults to 1e-6.
            line_search (str, optional): Whether to perform a line search. Defaults to "none".
              - "none" Does not perform a line search.
              - "exact" Perform an exact line search.
              - "inexact" Perform an inexact line search

            rho (float, optional): rho parameter in inexact line search, ignored for other types of line search. Defaults to 0.01.
            sigma (float, optional): rho parameter in inexact line search, ignored for other types of line search. Defaults to 0.1.
        """
        self.problem = problem
        self.stop_threshold = stop_threshold
        self.line_search = line_search
        self.rho = rho
        self.sigma = sigma
        self.name = "Generic optimizer"
        pass

    def exact_line_search(self, x, s):
        """Performs an exact line search at the point x, in the direction s.

        Args:
            - x (numpy array): The vector x as a 1d numpy array
            - s (numpy array): The search direction as a 1d numpy array

        Returns:
            float: The multiplier alpha of the search direction
        """
        problem = self.problem

        def dphi(alpha):
            return np.dot(s, problem.gradf(x + alpha * s, **problem.kwargs))

        alpha_new = fsolve(dphi, 1)
        return alpha_new

    def setup_inexact_line_search(
        self, f_: float, rho=0.01, sigma=0.1, tau2=1 / 10, tau3=1 / 2, alpha_init=0.1
    ):
        """Setup the inexact line search parameters

        Args:
            f_ (float): A lower bound on f, for which any value below this bound are to be accepted
            rho (float, optional): Controls the right bound for acceptable values of alpha. 0<rho<1/2. Defaults to 0.01.
            sigma (float, optional): Two sided test on the slope. The lower the value, the more accurate the line search. rho<=sigma<1 Defaults to 0.1.
            tau2 (float, optional): Controls the left bound in  which values of alpha is chosen for testing after the bracketing phase. Defaults to 1/10.
            tau3 (float, optional): Controls the right bound in  which values of alpha is chosen for testing after the bracketing phase.. Defaults to 1/2.
            alpha_init (float): Initial guess to use for the value of alpha
        """
        if rho < 1 / 2 and rho > 0:
            self.rho = rho
        else:
            raise ValueError(
                f"Invalid value of rho, a value between 0 and 1/2 is expected, got {rho}"
            )
        if sigma >= rho and sigma < 1:
            self.sigma = 0.1
        else:
            raise ValueError(
                f"Invalid value of sigma, a value between rho and 1 is expected, got {sigma}"
            )

        self.f_ = float(f_)
        self.tau2 = float(tau2)
        self.tau3 = float(tau3)
        self.alpha_init = alpha_init

    def inexact_line_search(self, x: npt.ArrayLike, s: npt.ArrayLike):
        """Performs an inexact line search at the point x, in the direction s.
        Based on the algorithm in Fletscher, Practical Optimization, 2nd Edition 2013, p37-39

        Args:
            - x (numpy array): The vector x as a 1d numpy array
            - s (numpy array): The search direction as a 1d numpy array

        Returns:
            float: The multiplier alpha of the search direction

        The line search uses some hyperparameters. by default,the rho and sigma parameters are set to 0.01 and 0.1
        """

        problem = self.problem
        try:
            sigma = self.sigma  # 0.9 for weak, 0.1 for fairly accurate line search?
            rho = self.rho  # p30
            tau2 = self.tau2  # p36
            tau3 = self.tau3  # p36
            f_ = self.f_
            alpha_init = self.alpha_init
        except Exception:
            raise Exception(
                "Please first initialize the inexact line search with the setup_inexact_line_search method"
            )

        def dphi(alpha):
            return np.dot(s, problem.gradf(x + alpha * s, **problem.kwargs))

        def phi(alpha, **kwargs):
            return problem.f(x + alpha * s, **kwargs)

        f0 = problem.f(x, **problem.kwargs)
        if(f_ > f0):
          print("Set lower bound of f is higher than current guess, returning as is...")
          return 0
        df0 = dphi(0)
        if(df0 > 0):
            print("Wrong search direction detected.")
        mu = (f_ - f0) / (rho * df0)
        alpha_new = min(alpha_init, mu)  # 0 < a_1 <= mu? #p37
        alpha = alpha_new  # p37
        alpha_old = 0
        B = 0
        for i in range(10):
            fa = problem.f(x + alpha * s, **problem.kwargs)
            if fa <= f_:
                break
            fa_old = problem.f(x + alpha_old * s, **problem.kwargs)
            dfa = dphi(alpha)
            if (fa > f0 + alpha * rho * df0).any() or (fa >= fa_old).any():
                a = alpha_old
                b = alpha
                B = 1
                break
            if (abs(dfa) <= -sigma * df0).any():
                break
            if (dfa >= 0).any():
                a = alpha
                b = alpha_old
                B = 1
                break
            if (mu <= 2 * alpha - alpha_old).any():
                alpha_new = mu
            else:
                alpha_new = (
                    2 * alpha - alpha_old
                )  # in [2 * alpha - alpha_old, min(mu, alpha + tau1 * (alpha - alpha_old))?
            alpha_old = alpha
            alpha = alpha_new
        if B == 1:
            a_new = a
            b_new = b
            for i in range(20):
                a = a_new
                b = b_new
                alpha = a + tau2 * (
                    b - a
                )  # in [a + tau2 * (b - a), b - tau3 * (b - a)]
                fa = problem.f(x + alpha * s, **problem.kwargs)
                f_a = problem.f(x + a * s, **problem.kwargs)
                if (fa > f0 + rho * alpha * df0).any() or (fa >= f_a).any():
                    a_new = a
                    b_new = alpha
                else:
                    dfa = dphi(alpha)
                    if (abs(dfa) <= -sigma * df0).any():
                        break
                    a_new = alpha
                    if (b - a) * dfa >= 0:
                        b_new = a
                    else:
                        b_new = b
        return alpha

    def step(self, x:npt.ArrayLike):
        """Performs a step for the optimizer

        Args:
            x (npt.ArrayLike): The current guess for x.

        Returns:
            x (nd.array), stop (bool): x is the new guess, Stop is true if the stopping criterion is hit
        """
        s = self.calculate_s()  # Search direction
        if self.line_search == "none":
            alpha = 1
        elif self.line_search == "exact":
            alpha = self.exact_line_search(x, s)
        elif self.line_search == "inexact":
            alpha = self.inexact_line_search(x, s)
        x = x + alpha * s
        stop_failure = alpha == 0
        stop_success = np.linalg.norm(s) < self.stop_threshold
        return x, stop_success, stop_failure

    @abstractmethod
    def calculate_s(self):
        """
        Method that calculates the search direction
        
        Returns:
            s (nd.array): Vector of the search direction
        """
        pass

    def solve(self, x0:npt.ArrayLike, max_iter=20):
        """Solve the optimization problem with the chosen optimization algorithm

        Args:
            x0 (nd.array): A 1d numpy vector with the initial guess for the solution
            max_iter (int, optional): Maximum number of iterations to do before giving up. Defaults to 20.

        Returns:
            _type_: _description_
        """
        x = x0
        self.xhist = [x]
        for i in range(max_iter):
            x, stop_success, stop_failure = self.step(x)
            self.xhist.append(x)
            if stop_success:
                self.success = True
                return x
            elif stop_failure:
                print("Stopping prematurely as the solver will not converge further")
                self.success = False
                return x
        self.success = False
        print("Optimizer did not converge")
        return x

    def get_fhist(self):
        f = self.problem.f
        self.fhist = []
        for j in range(len(self.xhist)):
            self.fhist.append(f(self.xhist[j],**self.problem.kwargs))
        return self.fhist
    
    def get_name(self):
        """Get the name of the optimizer

        Returns:
            str: The name of the optimizer
        """
        return "Unknown optimizer"
    
    def summary(self,sort_x = False):
        name = self.get_name()
        self.get_fhist()
        print(f"\n--- Optimization with {name}. Line search: {self.line_search} ---")
        try:
            s = "Success" if self.success else "No success"
            print(f"{s} after {len(self.xhist)} iterations")
        except AttributeError:
            pass
        print(f"Initial value of f: {self.fhist[0]}")
        print(f"Last value of f: {self.fhist[-1]}")
        if(len(self.xhist[-1])<20): #Dont print if too long
            lastx = self.xhist[-1]
            if(sort_x):
                lastx = np.sort(lastx)
            print(f"Last value of x: {lastx}")
            
            
        
        

class NewtonOptimizer(Optimizer):

    def calculate_s(self):
        problem = self.problem
        x = self.xhist[-1]
        hessian = finite_difference(problem.gradf, x, problem.epsilon, **problem.kwargs)
        grad = problem.gradf(x, **problem.kwargs)
        s = -np.linalg.solve(hessian, grad)
        return s
    
    def get_name(self):
        return "Newton"


class QuasiNewtonOptimizer(Optimizer):
    def __init__(
        self, problem, stop_threshold=1e-6, line_search="none", hessian_init="identity"
    ):
        """Generic class for a Quasi Newton optimizer.

        Args:
            problem (OptimizationProblem): Optimization problem to solve for
            stop_threshold (float, optional): Stopping threshold for the iterations. Defaults to 1e-6.
            line_search (str, optional): Type of line search performed. Defaults to "none".
              - "none": No line search.
              - "exact": Exact line search.
              - "inexact": Inexact line search.
            hessian_init (str, optional): How to initialize the inverse of the Hessian. Defaults to "identity".
              - "identity": Initialize with the identity matrix.
              - "fd" or "finite_diff": Initialize the Hessian with a finite difference approximation.
        """
        super().__init__(problem, stop_threshold, line_search)

        # Define how to initialize the hessian
        self.hessian_init = hessian_init
        pass

    def calculate_s(self):
        try:  # If H already exists
            H = self.H
            xnew = self.xhist[-1]
            x = self.xhist[-2]
            self.gnew = self.problem.gradf(xnew, **self.problem.kwargs)
            self.H = self.update_H(H, self.gnew, self.g, xnew, x)
        except AttributeError:  # In the case of the first step, do the usual stuff
            xnew = self.xhist[-1]
            self.gnew = self.problem.gradf(xnew, **self.problem.kwargs)
            self.init_H()
        s = -self.H @ self.gnew
        self.g = np.copy(self.gnew)
        return s

    def init_H(self):
        xnew = self.xhist[-1]
        if self.hessian_init == "finite_diff" or self.hessian_init == "fd":
            hessian = finite_difference(
                self.problem.gradf, xnew, self.problem.epsilon, **self.problem.kwargs
            )
            #Symmetrizing step
            hessian = 1/2 * (hessian + hessian.T)
            self.H = np.linalg.inv(hessian)
        elif self.hessian_init == "identity":
            self.H = np.identity(len(self.gnew))
        else:
            raise ValueError(
                "Unknown hessian initialization option: " + self.hessian_init
            )

    @abstractmethod
    def update_H(self, H, gnew, g, xnew, x):
        """
        Method that updates the estimate of H, the inverse of the Hessian function
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective function at the previous point x
        - gnew: gradient of objective function at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        pass
    
        
    def get_name(self):
        return "Unknown Quasi-Newton"


class GoodBroyden(QuasiNewtonOptimizer):
    
    
    def update_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective funtion at the previous point x
        - gnew: gradient of objective funtion at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x  # displacement between the new point xnew and the old point x
        y = (
            gnew - g
        )  # difference between in gradients between the new point xnew and the old point x
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = (
            H + (d - H @ y) / (d.T @ H @ y) @ d.T @ H
        )  # update Inverse Hessian by Sherman-Morisson formula
        return Hnew

    def get_name(self):
        return "Good Broyden"

class BadBroyden(QuasiNewtonOptimizer):

    def update_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective function at the previous point x
        - gnew: gradient of objective function at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x  # displacement between the new point xnew and the old point x
        y = (
            gnew - g
        )  # difference between in gradients between the new point xnew and the old point x
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d - H @ y) / (y.T @ y) @ y.T  # update

        return Hnew


    def get_name(self):
        return "Bad Broyden"


class SymmetricBroyden(QuasiNewtonOptimizer):
    def update_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective function at the previous point x
        - gnew: gradient of objective function at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x  # displacement between the new point xnew and the old point x
        y = (
            gnew - g
        )  # difference between in gradients between the new point xnew and the old point x

        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        u = (
            d - H @ y
        )  # difference between displacement d and the predicted change based on the current Hessian approximation
        a = 1 / (
            u.T @ y
        )  # the reciprocal the dot product of the vector u with the gradient change y

        Hnew = H + a * u @ u.T  # update

        return Hnew


    def get_name(self):
        return "Symmetric Broyden"

class DFP(QuasiNewtonOptimizer):
    
    
    def update_H(self, H, gnew, g, xnew, x):
        """
        # Parameters:

        - H: Inverse Hessian
        - g: gradient of objective funtion at the previous point x
        - gnew: gradient of objective funtion at the new point xnew
        - x: The current/previous point in the parameter space before the update
        - xnew: The new point in the parameter space
        """
        d = xnew - x  # displacement between the new point xnew and the old point x
        y = (
            gnew - g
        )  # difference between in gradients between the new point xnew and the old point x
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        Hnew = H + (d @ d.T) / (d.T @ y) - (H @ y @ y.T @ H) / (y.T @ H @ y)  # update
        return Hnew


    def get_name(self):
        return "DFP"

class BFGS(QuasiNewtonOptimizer):

    def update_H(self, H, gnew, g, xnew, x):
        d = xnew - x  # displacement between the new point xnew and the old point x
        y = (
            gnew - g
        )  # difference between in gradients between the new point xnew and the old point x
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        # dTy = d.T @ y
        # dyT = d @ y.T

        Hnew = (
            H
            + (1 + (y.T @ H @ y) / d.T @ y) * (d @ d.T) / d.T @ y
            - (d @ y.T @ H + H @ y @ d.T) / (d.T @ y)
        )

        return Hnew


    def get_name(self):
        return "BFGS"

class CompareBFGS(QuasiNewtonOptimizer):

    # Overloading the init to precise we compute the Hessian
    def __init__(
        self, problem, stop_threshold=1e-6, line_search="none", hessian_init="identity"
    ):
        super().__init__(problem, stop_threshold, line_search, hessian_init)
        self.Hhist = []
        self.HestHist = []
        pass

    def update_H(self, H, gnew, g, xnew, x):
        d = xnew - x
        y = gnew - g
        d = np.reshape(d, (d.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        dTy = d.T @ y
        dyT = d @ y.T

        Hnew = (
            H
            + (1 + (y.T @ H @ y) / dTy) * (d @ d.T) / dTy
            - (dyT @ H + H @ y @ d.T) / (dTy)
        )
        self.HestHist.append(Hnew)
        hessian = finite_difference(
            self.problem.gradf, xnew, self.problem.epsilon, **self.problem.kwargs
        )
        self.Hhist.append(hessian)
        return Hnew


    def get_name(self):
        return "BFGS"

if __name__ == "__main__":

    def g(x, r):
        return np.sum(r * x**2)

    def grad_g(x, r):
        return r * 2 * x

    epsilon = 1e-6
    pb = OptimizationProblem(g, gradf=grad_g, r=2)
    optimizer = DFP(pb, 1e-9, "inexact", "fd")
    optimizer.setup_inexact_line_search(0)
    optimizer.solve(np.array([2, 4]), 15)
    optimizer.summary()
    # print(f'optimizer.xhist: {optimizer.xhist}')
    # print(optimizer.HestHist)
    # print(optimizer.Hhist)
