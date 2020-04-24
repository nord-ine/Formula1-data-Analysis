import numpy as np
import time

class LinearModel():

    def __init__(self,metric):
        self.metric = metric
        
    
    def build_model(self,data,labels,Niter):
        self.hist=[]
        weights, self.hist = self.gradient_descent(data,np.random.rand(data.shape[1]),labels, Niter,speed=0.01)
        self.weights = np.array(weights)
        return self.hist
    def loss(self,predicted_labels,labels):
        '''
        Cette fonction de perte renvoie l'une de ces trois : MSE,RMSE,MAE
        '''
        if(self.metric=='MSE'):
            return np.square(predicted_labels-labels).mean()
        elif (self.metric=='RMSE'):
            return  np.sqrt(np.square(predicted_labels-labels).mean())
        else: # MAE
            return np.absolute(predicted_labels-labels).mean()




    def gradient_descent(self,data,weights,labels,niter=100,speed=0.05,tol=0.00001,i=0):
        '''
        Weights est un vecteur n colonnes, data est une matrice m lignes n colonnes, les predictions sont un vecteur m colonnes
        '''
        # local variable to store the hist
        prediction = np.dot(data,weights)
        coeff = (1.0*speed)/data.shape[0]
        lossIter = self.loss(prediction,labels)
        for j in range(0,(weights.shape[0])):
            myLoss = 0
            for k in range(0,data.shape[0]):
                myLoss += (prediction[k]-labels[k])*data[k][j]
            print('Iteration {}, weights[0] = {}, weights[1] = {}, le cout est de {}\r'.format(i,weights[0],weights[1],myLoss)),
            weights[j] = (weights[j]-(1.0*coeff*myLoss))
        if lossIter < tol or i >= niter:
            return weights , self.hist
        self.hist.append(lossIter)
        #Fonction recursive, ce n'est peut-etre pas la maniere la plus elegante de le faire.
        return self.gradient_descent(data,weights,labels,niter,speed,tol,i+1)


    def predict(self,data):
        return np.dot(data,self.weights)

    