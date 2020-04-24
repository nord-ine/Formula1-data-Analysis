from DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np
from collections import Counter
from collections import defaultdict



class RandomForest():
    """
    a class that represents the random forest algorithm
    
    parameters are the same as the decisiontree class exept here we have :
        - number_trees
        - number_features_persplit : repersents the number of features to consider when making a split (should be less than the number of features) to avoid correlation between the decisiontrees
    """
    def __init__(self,number_trees,number_features_persplit,min_element_perSplit,max_depth=10,max_iteration=100):
        self.max_depth = max_depth
        self.min_element_perSplit=min_element_perSplit # minimum d'element dans un noeud passe en parametre dans le constructeur
        self.max_iteration = max_iteration
        self.number_trees = number_trees
        self.number_features_persplit = number_features_persplit
        self.trees = np.array([DecisionTreeClassifier(self.min_element_perSplit,self.max_depth,self.max_iteration) for i in range(self.number_trees)])

    def build_forest(self,X,Y):
        """
        Arguments:
            X the data as a numpy array 
            Y labels of X data as numpy vector 
        """
        def get_sample(self):
            """
            a function to get a sample for each tree from the input data 
            """
            N = np.shape(X)[0]
            indexes = np.random.choice(N,N,replace=True) # we take N sample by replacement from N samples (we take indexes of the input data to get the corresponding X,Y)
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------")
            return X[indexes] , Y[indexes]

        # we build many trees using different samples 
        for t in self.trees:
            t.max_features= self.number_features_persplit
            x,y = get_sample(self)
            t.build_tree(x,y)



    def predict(self,X):
        """
        a method used to predict the labels of new unlabeled data
        
        Returns:
        [list]  [a predicted classes for each data point(row)]
        """
        Predictions = []
        # each row reprensents a the predictions of a tree
        for i,t in enumerate(self.trees) :
            Predictions.append(t.predict(X))
        
        predicted_classes=[]
        for i in range(np.shape(Predictions)[1]):
            # we take the most common class predicted by the trees 
            predicted_classes.append(Counter([row[i] for row in Predictions]).most_common(1)[0][0])

        return predicted_classes