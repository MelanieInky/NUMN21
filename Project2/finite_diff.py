# CODE FOR THE FINITE DIFFERENCE FUNCTION


import numpy as np
from collections.abc import Callable
import numpy.typing as npt
from typing import Concatenate


DEBUG = False

def finite_difference(f: Callable[Concatenate[npt.ArrayLike,...],npt.ArrayLike],
                      x: npt.ArrayLike,
                      eps=1e-6,
                      **kwargs) -> npt.ArrayLike:
    """Computes the Jacobian of the function f at the point x via a first order, finite difference approach

    Args:
        f : A function R^n -> R^m for which to compute the derivative, of the form f(x,**kwargs)
        x (npt.ArrayLike): A vector in R^n
        eps (float, optional): finite difference value to use. Defaults to 1e-6.
        **kwargs : Other arguments to be passed to f
    Returns:
        ndarray: the Jacobian of f at the point x of dimension (R^n * R^m) 
    """
    x = np.array(x,dtype = float)
    x = np.atleast_1d(x)
    print(x)
    if not kwargs:
        fx = f(x)
    else:
        fx = f(x, **kwargs)
    if len(np.asarray(fx).shape) == 0:
        fx = np.array([fx])

    J = np.zeros((len(fx), x.shape[0]))
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        if not kwargs:
            fx_plus = f(x_plus)
        else:
            fx_plus = f(x_plus, **kwargs)
        if DEBUG:
            print(f"fx : {fx}")
            print(f"fx_plus: {fx_plus}")
            print((fx_plus - fx).shape)
        J[:, i] = (fx_plus - fx) / eps
    return J


if __name__ == "__main__":
    """
    Small test of the finite difference code
    """


    def f(x):
        return np.array([x[0], x[1] ** 2, x[0] ** 2 + x[1]])

    epsilon = 1e-6
    J = finite_difference(f, np.array([1, 2]))
    print(J)

    