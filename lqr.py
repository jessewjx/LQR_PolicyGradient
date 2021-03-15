"""
policy gradient methods for LQR
reference: https://arxiv.org/pdf/1801.05039.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from scipy import linalg as la

class LQR_Solver:
    def __init__(self,A,B,Q,R,d,k,numstep,step_size,thres):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.d = d
        self.k = k
        self.thres = thres
        self.numstep = numstep
        self.step_size = step_size
               

    def reward(self, x, K):
        cost = multi_dot([x.T,self.Q - multi_dot([K.T, self.R, K]),x]).item(0)          
        return cost
    
    def step(self,K, x):
        return np.dot((self.A - np.dot(self.B,K)) , x)

    
    def ARE(self):
        P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K_opt = multi_dot([np.linalg.inv(self.R + multi_dot([self.B.T, P, self.B])), self.B.T, P, self.A])
        
        return K_opt

    
    def ModelBased_PolicyGradient(self):
        P_k = self.Q
        K = np.random.rand(self.k,self.d)
        sigmak = np.zeros((self.d,self.d))
        num_rollout = 5
        
        for t in range(0,self.numstep):
            # incremental reward operator
            P_k = self.Q + multi_dot([K.T, self.R ,K]) +\
            multi_dot( [(self.A-np.dot(self.B,K)).T, P_k ,self.A-np.dot(self.B, K)] )

            # calculate policy gradient with state covariance matrix
            x,cost = self.RollOut(K,num_rollout)
            sigmak = np.dot(x,x.T)
            grad_C = 2*np.dot( np.dot(self.R+ multi_dot([self.B.T, P_k, self.B]),K)  - multi_dot([self.B.T,P_k,self.A]) ,sigmak) 
            
            # gradient descent
            K = K - self.step_size * grad_C
    
            if grad_C.any() <= self.thres:
                print("Model-based Policy Gradient converges...")
                break

        return K
  

    def ModelBased_NaturalPolicyGradient(self):
        P_k = self.Q
        K = np.random.rand(self.k,self.d)
        sigmak = np.zeros((self.d,self.d))
        num_rollout = 5
        
        for t in range(0,self.numstep):
            # incremental reward operator
            P_k = self.Q + multi_dot([K.T, self.R ,K]) +\
            multi_dot( [(self.A-np.dot(self.B,K)).T, P_k ,self.A-np.dot(self.B, K)] )

            # calculate natural policy gradient with state covariance matrix
            x,cost = self.RollOut(K,num_rollout)
            sigmak = np.dot(x,x.T)
            grad_C = 2*np.dot( np.dot(self.R+ multi_dot([self.B.T, P_k, self.B]),K)  - multi_dot([self.B.T,P_k,self.A]) ,sigmak) 
            K = K - self.step_size * np.dot(grad_C, np.linalg.inv(sigmak))
           
            
            if grad_C.any() <= self.thres:
                print("Model-based Natural Policy Gradient converges...")
                break

        return K
  

    def ModelFree_PolicyGradient(self, m, length, smoothp):
        K = np.random.rand(self.k,self.d)
        
        for t in range(0,self.numstep):
            est_gradC = np.zeros((self.d,self.d))
            est_sigmak = np.zeros((self.d,self.d))
            
            # Monte Carlo simulation for gradient estimation
            for i in range(0,m):
                # sample policy matrix from a uniform distribution
                U_i = np.random.normal(0, 1,(self.d,self.d))
                
                # simulate state,action trajectory for l steps 
                x,cost = self.RollOut(K+U_i,length)
                est_gradC  = est_gradC + sum(cost)*U_i
                est_sigmak = est_sigmak + np.dot(x,x.T)

            # estimate policy gradient with likelihood ratio
            est_gradC = est_gradC * self.d/(m*smoothp*smoothp)
            est_sigmak = est_sigmak/m

            # gradient descent
            K = K - self.step_size * est_gradC
                 
            if est_gradC.any() <= self.thres:
                print("Sample-based Policy Gradient converges...")
                break
                
        return K
        
        
    def RollOut(self,K,length):
        x = np.zeros((self.d,length))
        x[:,0] = np.random.uniform(low=-0.5, high=0.5,size=(self.d,))

        cost = []
        cost_t = self.reward(x[:,0],K)
        cost.append(cost_t)
      
        # evolution of state,action pair
        for t in range(1,length):
            x[:,t] = self.step(K, x[:,t-1])
            cost_t = cost_t + self.reward(x[:,t],K)
            cost.append(cost_t)
            
        return x,cost
            
        


