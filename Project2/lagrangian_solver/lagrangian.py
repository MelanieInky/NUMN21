from quadratures import quadrature
import numpy as np
import matplotlib.pyplot as plt

class LagrangianProblem:
    def __init__(self,K,V,grad_K,grad_V) -> None:
        """Class for hopefully solving physics problem with Hamilton principle
        
        The potential and kinetic energy functions  have the form f(x,x_dot,t):R^n -> R. 
        The gradients have the form f'(x,t):R^n -> R^n
        """
        self.K = K #Kinetic energy function
        self.V = V #Potential energy function
        self.grad_K = grad_K #Gradient of the kinetic energy function
        self.grad_V = grad_V #Gradient of the potential energy function
        def L(x,x_dot,t):
            return self.K(x,x_dot,t) - self.V(x,x_dot,t) #Lagrangian
        self.L = L
        def grad_L(x,x_dot,t):
            return self.grad_K(x,x_dot,t) - self.grad_V(x,x_dot,t)
        self.grad_L = grad_L
        pass
    
    def get_velocity(self,X_list,t_list):
        """
        Get the velocity vector at a specific point, using a finite difference formula,
        here, first order backward difference should do the trick, but this should be expendable

        Args:
            X_list (_type_): The positions vectors at the specified time in t_list
            t_list (_type_): Vector of times.
        """
        dt = t_list[1] - t_list[0]
        x_dot = (X_list[1] - X_list[0]) / dt
        return x_dot
        
        
    
    def S(self,x_list,x_dot_init, t1,t2,method = "simpson"): #Action
        """Compute the action

        Args:
            x_list (_type_): Array of size (k*n), k being the number of quadrature points
            x_dot_init: Array of size n, initial velocity vector
            t1 (_type_): Initial time
            t2 (_type_): Final time
        """
        if(method == "simpson"):
            w = np.array([1/6,4/6,1/6])
            c = np.array([0,1/2,1])
        else:
            raise NotImplementedError("Unknown quadrature method")
        
        
        x_dot_list = np.zeros_like(x_list)
        x_dot_list[0] = x_dot_init
        #Calculate the speed vector list first.
        t_list = t1 + c*t2
        for k in range(1,np.shape(x_list)[0]):
            x_dot_list[k] = self.get_velocity(x_list[k-1:k+1],t_list[k-1:k+1])
        self.action, _ = quadrature(self.L,x_list,x_dot_list,w,c,t1,t2,self.grad_L)
        return self.action
        

if __name__ == "__main__":
    m = 1
    g = 10
    def K(x,x_dot,t):
        return 1/2 *m* x_dot**2
    
    def V(x,x_dot,t):
        return m*g*x
    
    def gradK(x,x_dot,t):
        return 0
    
    def gradV(x,x_dot,t):
        return m*g
    
    
    test = LagrangianProblem(K,V,gradK,gradV)
    
    #Higher T makes for some bigger error.
    T = 0.05
    
    variations = np.linspace(-1,1,50)
    actions = np.zeros_like(variations)
    
    x_list = np.array([[0],[-1/2*g*(T/2)**2],[-1/2*g*T**2]])
    x_dot_init = np.array([0])
    
    #Variation in midpoint
    for i , eps in enumerate(variations):
        x_list_tmp = np.copy(x_list)
        x_list_tmp[1] += eps
        actions[i] = test.S(x_list_tmp,x_dot_init,0,T)
    
    plt.figure()
    plt.plot(variations,actions) #The minimum is clearly at 0 variation
    
    #Variation in endpoint
    for i , eps in enumerate(variations):
        x_list_tmp = np.copy(x_list)
        x_list_tmp[2] += eps
        actions[i] = test.S(x_list_tmp,x_dot_init,0,T)
    
    plt.figure()
    plt.plot(variations,actions) #The minimum is clearly at 0 variation
    pass
    