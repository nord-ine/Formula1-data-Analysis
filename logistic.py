import numpy as np

class LogisticClassifier():

    def __init__(self,threshold):
        self.threshold = threshold
        

    def sigmoid(self,x):
        '''
        Retourne la fonction sigmoide qui nous permet d'exprimer une probabilite entre 0 et 1
        '''
        return 1.0/(1+np.exp(-x))

    def propagate(self,w, b, X, Y):
        '''
        In : w (numpy array) de dimension (n,1) : les poids affectes aux donnees
            b (numpy array) de dimension (1,m) : le biais
            X (numpy array) de dimension (n,m) : les donnees avec n features et m examples
            Y (numpy array) de dimension (1,m) : les labels
        Out : grads {dw : np.array(n,1), db = np.array(1,m)} : dictionnaire avec la derivee de w et b pour pouvoir mettre a jour les poids
            cost (float) : le cout
        '''
        m = X.shape[1]
        A = self.sigmoid(np.dot(w.T, X)+b)
        cost = (-1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
        dw = (1.0/m)*np.dot(X,(A-Y).T)
        db = (1.0/m)*np.sum(A-Y)

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                "db": db}
        
        return grads, cost

    def optimize(self,w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        '''
        Boucle principale de l'algorithme
        In : w (numpy array) de dimension (n,1) : les poids affectes aux donnees
            b (numpy array) de dimension (1,m) : le biais
            X (numpy array) de dimension (n,m) : les donnees avec n features et m examples
            Y (numpy array) de dimension (1,m) : les labels
            num_interations (int) : nombre d'iterations de l'algorithme
            learning_rate (float) : vitesse de convergence
            print_cost (bool) : afficher ou pas le cout
        Out : params {w: np.array(n,1), b: np.array(1,m)} : les poids et le biais
            grads {dw: np.array(n,1), db: np.array(1,m)} : les derivees des poids et du biais
            costs (list(float)) : l'historique des couts
        '''
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w,b,X,Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate*dw
            b = b - learning_rate*db
            if i % 100 == 0:
                costs.append(cost)
            
        
        params = {"w": w,
                "b": b}
        
        grads = {"dw": dw,
                "db": db}
        
        return params, grads, costs


    def fit(self,X_train, Y_train, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        '''
        Creation du modele : on entraine sur le training test et on teste sur le test set (bah ouais logique)
        Retourne un dictionnaire avec toutes les informations
        '''
        self.w ,self.b = np.zeros((X_train.shape[0],1)),0
        parameters, grads, costs = self.optimize(self.w,self.b,X_train,Y_train,num_iterations=num_iterations,learning_rate=learning_rate,print_cost=print_cost)
        self.w = parameters["w"]
        self.b = parameters["b"]
        Y_prediction_train = self.predict(X_train).transpose()
        print("train accuracy: {} % \n".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        d = {"costs": costs,
            "Y_prediction_train" : Y_prediction_train, 
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}
        
        return d
    def predict(self,X):
        '''
        In : w (numpy array) de dimension (n,1) : les poids affectes aux donnees
            b (numpy array) de dimension (1,m) : le biais
            X (numpy array) de dimension (n,m) : les donnees avec n features et m examples
        Out : Y_prediction (numpy array) de dimension (1,m) : les labels predits par l'algorithme
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = self.w.reshape(X.shape[0], 1)
        A = self.sigmoid(np.dot(w.T,X)+self.b)
        for i in range(A.shape[1]):
            if A[0][i] > self.threshold:
                Y_prediction[0][i] = 1
            else:
                Y_prediction[0][i] = 0
        assert(Y_prediction.shape == (1, m))
        Y_prediction = Y_prediction.transpose()
        return Y_prediction