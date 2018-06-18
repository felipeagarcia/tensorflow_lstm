# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:02:31 2017

@author: Felipe Aparecido Garcia
@github: https://github.com/felipeagarcia/
"""
import numpy
import pickle
import random

#To understand everything in this code, it's recomended to read 
'''
Rabiner, Lawrence R. "A tutorial on hidden Markov models and selected applications in speech recognition."
Proceedings of the IEEE 77.2 (1989): 257-286.
    
'''
class hmm():
    "An implementation of hidden markov models based on Rabiner's book"
    def __init__(self, model_name, n, m, A = None, B = None, pi = None):
        'Initialize the model'
        if(A is None):
            A = {} #A is the trasition matrix
            self.A = self.initializeMatrix(A, n, n)
        else:
            self.A = A

        if(B is None):
            B = {} #B is the emission matrix
            self.B = self.initializeMatrix(B, n, m)
        else:
            self.B = B

        if(pi is None):
            pi = [] #pi is the initial states distribution
            self.pi = self.initializePi(pi, n)
        else:
            self.pi = pi

        #to create a model, we need to know some parameters:
        #n is the number os states
        #m is the number of possible observations
        self.name = model_name
        self.n = n
        self.m = m

        #The parameters above are part of the canonical problems
        self.alfa = {}
        self.beta = {}
        self.eta = {}
        self.gama = {}
        self.c = {}
        
    def initializeMatrix(self,Matrix, n, m):
        Matrix = numpy.zeros((n,m))
        if m % 2 == 0:
            for i in range(0,n):
                aux =abs(random.uniform(0.001, 0.01))
                for j in range(0, m):
                    Matrix[i][j] = (1.0)/m + aux

                    if(aux > 0):
                        aux = -1 *aux
                        if abs(aux) >= (1.0/m) -0.01:
                            aux = aux/100
                    else:
                        aux = -1*aux*j
                        if abs(aux) >= (1.0/m) -0.01:
                                aux = aux/100
        else:
            for i in range(0,n):
                aux = abs(random.uniform(0.001, 0.01))
                for j in range(0, m):
                    if j==0:
                        Matrix[i][j] = 1.0/ m
                    else:
                        Matrix[i][j] = (1.0)/m + aux
                        if(aux > 0):
                            aux = -1 *aux
                            if abs(aux) >= (1.0/m) -0.01:
                                aux = aux/100
                        else:
                            aux = -1*aux*j
                            if abs(aux) >= (1.0/m) -0.01:
                                aux = aux/100
        return Matrix

    def initializePi(self, pi, n):
        pi = []
        if n % 2 == 0:
            aux = abs(random.uniform(0.001, 0.01))
            for j in range(n):
                pi.append ( (1.0)/n + aux)
                if(aux > 0):
                    aux = -1 *aux
                    if abs(aux) >= (1.0/n) -0.01:
                        aux = aux/100
                else:
                    aux = -1*aux*j
                    if abs(aux) >= (1.0/n) -0.01:
                        aux = aux/100
        else:
        
            aux = abs(random.uniform(0.001, 0.01))
            for j in range(n):
                if j==0:
                    pi.append ( 1.0/ n)
                else:
                    pi.append( (1.0)/n + aux)
                    if(aux > 0):
                        aux = -1 *aux
                        if abs(aux) >= (1.0/n) -0.01:
                            aux = aux/100
                    else:
                        aux = -1*aux*j
                        if abs(aux) >= (1.0/n) -0.01:
                            aux = aux/100
        return pi
        
    def getName(self):
        return self.name
    
    def getA(self):
        return self.A.copy()
    
    def getB(self):
        return self.B.copy()
    
    def getPi(self):
        return self.pi.copy()
    
    def setA(self, A):
        self.A = A.copy()
        
    def setB(self, B):
        self.B = B.copy()
        
    def setPi(self, pi):
        self.pi = pi.copy()
    
    def normalize(self, Matrix, n, m):
        'make the sum of a row in a matrix be one'
        for i in range(0,n):
            aux = 0
            for j in range (0,m):
                aux+= Matrix[i][j]
            for j in range(0,m):
                if aux > 0:
                    Matrix[i][j] = Matrix[i][j]/aux
                else:
                    print("Fatal error")
                    

        return Matrix
    
    def normalize_array(self, array, n):
        aux = 0
        for i in range(n):
            aux += array[i]
        for i in range(n):
            array[i] = array[i]/aux 
        return array
        
    def foward_scaled(self, O, t):
        'the foward algorithm'
        c = []
        alfa = []

        #####
        A = self.getA()
        B = self.getB()
        n = getattr(self, 'n')
        pi = self.getPi()
        
        for i in range(t):
            c.append(0)
            
        for i in range(t):
            alfa.append([])
            for j in range(n):
                alfa[i].append(0)
        ####### 
        
        for i in range(n):
            alfa[0][i] = pi[i] * B[i][O[0]]
            c[0] += alfa[0][i]
        if c[0] > 0:
            c[0] = 1/c[0]
        else:
            print("Critical error, alfa = 0")

        for i in range(n):
            alfa[0][i] = c[0] * alfa[0][i]

        for i in range(0,t-1):
            
            for j in range(0,n):
                for h in range (0,n):
                    alfa[i+1][j] += alfa[i][h]* A[h][j]
                alfa[i+1][j] = alfa[i+1][j]*B[j][O[i+1]]
                c[i+1] += alfa[i+1][j]
            if c[i+1] > 0:
                c[i+1] = 1/c[i+1]
            else:
                print("Error, c <= 0")
            for j in range(0,n):
                alfa[i+1][j] = c[i+1] * alfa[i+1][j]
        return alfa,c
    
    def backward_scaled(self, O, t):
        'The backward algorithm'
        n = getattr(self, 'n')
        A = self.getA()
        B = self.getB()
        c = getattr(self, 'c')
        beta = numpy.zeros(( t, n))
        
        for i in range(0, n):
            beta[t-1][i] = c[t-1]
        
        for i in range(t-2,-1,-1):
            for j in range (0 , n):
                for h in range(0,n):
                    beta[i][j] += A[j][h] * B[h][O[i+1]] * beta[i + 1][h]
                beta[i][j] = c[i] * beta[i][j]
        return beta
    
    def compute_prob(self, O):
        'compute the log_10 of P(O|model)'
        t = len(O)
        self.alfa, self.c = self.foward_scaled(O,t)
        logProb = abs(sum([numpy.log10(c_aux) for c_aux in self.c]))
        return logProb
    
    def compute_eta_gama(self, O, t):
        'computing eta and gama wich we will use to reestimate parameters'
        A = self.getA()
        B = self.getB()
        n = getattr(self, 'n')
        alfa = getattr(self, 'alfa')
        beta = getattr(self, 'beta')
        aux = 0.0
        eta = numpy.zeros((t,n,n))
        gama = numpy.zeros((t,n))
        for i in range(0,t-1):
            aux = 0.0
            for j in range(0,n):
                for h in range(0,n):
                    aux += alfa[i][j]*A[j][h]*B[h][O[i+1]]*beta[i+1][h]
            for j in range(0,n):                   
                for h in range(0,n):
                    if aux > 0.0: 
                        eta[i][j][h] = (alfa[i][j]*A[j][h]*B[h][O[i+1]]*beta[i+1][h])/aux
                    else:
                        print("Error, eta <= 0!")
                    gama[i][j] += eta[i][j][h] 
        #special case for t-1               
        aux = 0
        for i in range(0,n):
            aux += alfa[t-1][i]
        for i in range(0,n):
            if aux > 0:
                gama[t-1][i] = (alfa[t-1][i]/aux)            
            else:
                print("Error, alfa = 0")      
        return eta, gama
    
    def computeA(self, t, min_val):
        'reestimates A'
        A = self.getA()
        n = getattr(self, 'n')
        eta = getattr(self, 'eta')
        gama = getattr(self, 'gama')
        for i in range(0,n):
            for j in range(0,n):
                aux1, aux2,  = 0.0, 0.0
                
                for z in range(0, t-1):
                    aux1 += eta[z][i][j]
                    aux2 += gama[z][i]
                   
                if aux2 > 0.0 :
                    A[i][j] = (aux1/ aux2)
                    if A[i][j] <= min_val:
                        A[i][j] = (min_val)
                else:
                    print("Error")
        A = self.normalize(A,n,n)
        return A
    
    def computeB(self, O, t, min_val):
        'reestimates B'
        B = self.getB()
        n = getattr(self, 'n')
        m = getattr(self, 'm')
        gama = getattr(self, 'gama')
        for i in range(0,n):
            for j in range(0, m):
                aux1, aux2 = 0.0, 0.0
                
                for z in range(0, t):
                    if O[z] == j:
                        aux1 += gama[z][i]
                    aux2 += gama[z][i]
                   
                if aux2 > 0:
                    B[i][j] = (aux1/aux2)
                    if B[i][j] <= min_val:
                        B[i][j] = (min_val)
                else:
                    print("Error")
        B = self.normalize(B, n, m)
        return B
    
    def computePi(self, min_val):
        'reestimates pi'
        n = getattr(self, 'n')
        pi = self.getPi()
        gama = getattr(self, 'gama')
        for i in range(0,n):
           pi[i] = (gama[0][i])
           if pi[i] <= min_val:
                pi[i] = (min_val) #probabilities cant be zero
        pi = self.normalize_array(pi,n) #we need this to assure that the sum of probs is one
        return pi

    def train(self, O):
        'the Baum-Welch algorithm'
        #@arg O: observations
        min_val = 0.0000000001
        count = 0
        max_count = 1000 #number of iterations
        tempA = {}
        tempB = {}
        tempPi = {}
        t = len(O)
        while  count < max_count:
            self.alfa, self.c = self.foward_scaled(O,t)
            old_prob = self.compute_prob(O)
            #print("log(P(O/lambda)) = " + str(old_prob))
            self.beta = self.backward_scaled(O, t)

            self.eta, self.gama = self.compute_eta_gama(O, t)
            #reestimating parameters
            tempPi = self.computePi(min_val)
            tempA = self.computeA(t,min_val)
            tempB = self.computeB( O, t, min_val)

            #now we need to guarantee that the models has improved, otherwise, we discard the changes
            auxA = self.getA()
            auxB = self.getB()
            auxPi = self.getPi()

            self.setA(tempA)
            self.setB(tempB)
            self.setPi(tempPi)

            totalProb = self.compute_prob(O)
            print("log(P(O|lambda):", totalProb, self.name, count)
            if(old_prob > totalProb):
                #the model has improved, we keep the changes
                count = count + 1
            else:
                #the model didn't improved, now we discard the reestimated parameters and stop the reestimation process
                self.setA(auxA)
                self.setB(auxB)
                self.setPi(auxPi)
                break
        #serializing the model to a file
        with open(self.name, 'wb+') as fp:
            pickle.dump(self, fp)