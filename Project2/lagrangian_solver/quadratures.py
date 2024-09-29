"""
Quadrature functions that should be useful.
"""
import numpy as np


def quadrature(f,x_list,x_dot_list, w,c,t1,t2,grad_f = None):
    """Generic formula to estimate the integral from t1 to t2 of f(x,t)

    Args:
        - f (_type_): The function f(x,x_dot,t)
        - x_list: A list with elements being the values of x at the quadrature points
        - x_dot_list: A list with elements being the values of x_dot, at the quadrature points
        - w (_type_): The associated weights
        - c (_type_): Vector used to offset the time values
        - t1 (_type_): Lower time bound of the integral
        - t2 (_type_): Upper time bound of the integral
        - grad_f: Gradient of f w.r.t x, return the gradient w.r.t the x_list coordinates as well,
        if specified.
        
        
    The function is written in such a way to facilitate treating the values of x as unknown.
    
    Example w and c:
    - Midpoint: w = 1, c = 1/2
    - Trapezoidal w = [1/2,1/2], c = [0,1]
    - Simpson rule  w = [1/6, 4/6, 1/6], c = [0,1/2,1]
    """
    h = t2-t1
    w = np.squeeze(w) #making sure there are no empty dimensions
    c = np.squeeze(c)
    
    k = x_list.shape[0]
    n = x_list.shape[1]
    ## Input verifications
    if(np.shape(x_dot_list) != (k,n)):
        raise Exception("x and xdot lists should have the same dimensions")
    if(len(w.shape) != 1 or len(c.shape) != 1):
        raise Exception("w and c should be one dimensional")
    if(w.shape[0] != k or c.shape[0] != k):
        raise Exception("w and c should have the same lengths equal to the one of x_list")
    if(grad_f is not None):
        grad_quad = np.zeros(k*n)
    
    quad = 0
    for i in range(k):
        t = t1 + c[i]*h
        quad+= w[i] * f(x_list[i],x_dot_list[i],t)
        if(grad_f is not None):
            grad_quad[n*i:n*(i+1)] = w[i] * grad_f(x_list[i],x_dot_list[i],t)
    
    quad *= h
    if(grad_f is not None):
        grad_quad *= h
        return quad, grad_quad
    return quad , None




if __name__=='__main__':
    def f(x,x_dot,t):
        return t
    x_list = np.zeros((2,3))
    x_dot_list = np.copy(x_list)
    w = [1/2,1/2]
    c = [0,1]
    test, _= quadrature(f,x_list,x_dot_list,w,c,1,2)
    print(f'Testing trap rule: exp: 1.5, got = {test}') #Should be 1.5
    
    def g(x,x_dot,t):
        return t**2

    #Simpson rule should integrate g exactly
    t1 = -2
    t2 = 4
    x_list = np.zeros((3,12))
    x_dot_list = x_list.copy()
    w = [1/6,4/6,1/6]
    c = [0,0.5,1]
    test , _ = quadrature(g,x_list,x_dot_list,w,c,t1,t2)
    print(f'Testing simpson rule: exp: {t2**3/3-t1**3/3}, got = {test}')
    
    def h(x,x_dot,t):
        return np.sum(x**2) + t
    
    def grad_h(x,x_dot,t):
        return 2*x
    
    n = 12
    x_list = np.arange(3*n).reshape((3,n))
    grad_expected = 2*x_list.flatten()
    grad_expected = grad_expected *  np.repeat([1/6,4/6,1/6],n)
    t1 , t2 = 0,1
    test , grad_test = quadrature(g,x_list,x_dot_list,w,c,t1,t2,grad_h)
    diff_norm = np.linalg.norm(grad_expected-grad_test)
    print(f"Testing the gradient, norm of the diff between expected and computed: {diff_norm}")
    
    
    
    
