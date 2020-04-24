# coding: utf-8
import numpy as np
from collections import Counter
from collections import defaultdict


class DecisionTreeClassifier():
    """
    [class that represents the Decision tree algo]
         min_element_inNode : min number of elements that a node can have
         max_depth : maximum depth of the tree 
         max_iteration : maximum number of iteration to get a split that has a positive gain of information
         max_features : equal zero when this class is used as a decision tree algorithm or equal to a value passed as a parameter when used in random forest
    """

    def __init__(self,min_element_inNode,max_depth=10,max_iteration=100):
        self.max_depth = max_depth
        self.min_element_inNode=min_element_inNode # minimum d'element dans un noeud passe en parametre dans le constructeur
        self.max_iteration = max_iteration
        self.max_features = 0 

    def build_tree(self,X,Y):
        """
        it's a method to create the tree by calling recursivly the method 'extend_tree'
        it takes two parameteres
        X :  the data as a numpy array 
        Y : labels of X data as numpy vector 
        """
        self.number_classes= len(set(Y)) 
        self.number_features = np.shape(X)[1] 
        self.root = self.extend_tree(X,Y,None,0,1) # root is the first node , parent_node=None, depth = 0 , parent_info_gain = maxvalueOFGini=1

    def extend_tree(self,x,y,parent_node,depth,parent_info_gain):

        """
        [a recursive method that builds the tree ]
        parameter:
        x : the data at the current node
        y : labels of x
        parent_node : predecesor of the curent node 
        depth : the current length of the tree
        parent_info_gain : information gain of the predecesor
        Returns:
            [node] at each iteration we create a node , at the end we will have a tree  with the nodes that we previously created
        """

        number_samples = np.shape(x)[0] 
        current_node = Node(parent_node)

        if len(set(y)) == 1 :  # check if we only have one type of class in the current node
            current_node.leaf=True # the node is a leaf 
            current_node.predicted_class = y[0] # the predicted class is the first one since they are all the same 
            print(y)
            print(len(y))

        # we check if the depth is less than the maximum_depth and the number of sample have to be greater than 2 times the minimun sample per node because if we split it we will get nodes with a number of elements less than what is required :
        elif depth<self.max_depth and number_samples>2*self.min_element_inNode:   

            feature_index ,split_value,node_gini = self._get_split(x,y,number_samples,parent_info_gain) # call to get_split method that return the feature and the value of the current split

            if feature_index==None : # if no split that increases information gain is found 
                current_node.leaf=True
                current_node.predicted_class = (Counter(y).most_common(1))[0][0] # we count the number of occurences of each class and we choose the most common one
                print(y)
                print(len(y))
                return current_node # we stop the recursion and we return the current node 

            current_node.feature_index, current_node.split_value = feature_index , split_value 
            indexes_left,indexes_right = [],[]
            for i,element in enumerate(x): # iterate over each element in the current node and check if it's value in the chosen feature is greater(left_node) or smaller(right_node) and save its index to later get the x,y of left and right children 
                if element[feature_index]>split_value:
                    indexes_left.append(i)
                else:
                    indexes_right.append(i)
            x_left , y_left = x[indexes_left], y[indexes_left]
            x_right, y_right = x[indexes_right], y[indexes_right]
            depth+=1
            current_node.left=self.extend_tree(x_left,y_left,current_node,depth,node_gini) # recursive call to extend_tree method  
            current_node.right=self.extend_tree(x_right,y_right,current_node,depth,node_gini)
        else : # if the first condition is not met (depth , number of samples of the current node )
            current_node.leaf=True #
            current_node.predicted_class = (Counter(y).most_common(1))[0][0] # right a formula to calculte the majority class in this node 
            print(y)
            print(len(y))
        return current_node

    






    def _get_split(self,x,y,number_samples,parent_info_gain):
        """
        a method responsible of finding a random split that increases the gain of information 
        
        Arguments:
            x {[array]} -- the data at the current node
            y {[vector]} --  labels of x
            number_samples {[int]} -- number of samples in the current node 
            parent_info_gain {[int]} -- gini of parent node to compare it to the gini resulted from the split 
        
        Returns:
            [feature] -- the feature in which the split get an information gain 
            [split_value] -- 
            information gain
        """
        def get_gini(dict,nb):
            """
            a function to compute the gini of each side of the split 
            
            Arguments:
                dict {[dictionnary]} -- dictionnary containing each class with its number of occurences
                nb {[int]} -- [description]
            
            Returns:
                gini
            """
            sum=0
            for d in dict.keys():
                sum+=np.power((dict[d]/nb),2)
            return 1-sum 

        feature = 0 # initialisation
        split_value = 0
        k_iteration=0
        gain_information = 2 # initialising gini to a value>1  to enter the while loop
        while(parent_info_gain-gain_information<0):
              
            if  k_iteration>self.max_iteration : # break from the loop if the number of iterations is greater than the max(passed as a parameter) and we returns none 
                return None,None,None
            feature = self._get_randomFeature() # get a random feature from the get_randomFeature method
            f_vector = x[:,feature] # the column vector of this feature
            min_vector = min(f_vector)
            max_vector = max(f_vector)
            split_value = np.random.uniform(min_vector+(min_vector/10),max_vector-(max_vector/10)) # get a  random split_value 

            d_left= defaultdict(lambda: 0) # a python dictionnary that initialise the number of occurence of each class to zero 
            d_right= defaultdict(lambda: 0) 
            nb_left=0 # number of elements in the left dictionnary
            nb_right = 0

            for i,element in enumerate(x): 
                if element[feature] > split_value :
                    d_left[y[i]]+=1
                    nb_left+=1
                else:
                    d_right[y[i]]+=1
                    nb_right+=1

            if nb_left>self.min_element_inNode and nb_right>self.min_element_inNode: 
                gain_information = ((nb_left/number_samples)*get_gini(d_left,nb_left))+((nb_right/number_samples)*get_gini(d_right,nb_right)) # 

            k_iteration+=1
        return feature,split_value,gain_information


    """
    a method that returns a random features from :
    - the whole list of features (when this class is used as a decision tree algorithm)
    - or from a subset of the list of features to avoid correlation when this class is used in the random forest algorithm
    Returns:
    [ a random feature]
     """
    def _get_randomFeature(self):
        if self.max_features==0 :
            return np.random.randint(0,self.number_features)
        else :
            f = np.random.choice(self.number_features,self.max_features,replace=False)
            return np.random.choice(f,1)




    def predict(self,X):
        """
        a method used to predict the labels of new unlabeled data
        
        Returns:
        [list]  [apredicted classes for each data point(row)]
        """
        def predict_xi(x):
            """
            a function used to get the predicted class of each data point of 'X' (a row)
            
            
            Returns:
                [the predicted class of x]
            """
            node = self.root
            while(not (node.leaf)):
                if x[node.feature_index]>node.split_value:
                    node = node.left
                else:
                    node = node.right
                
            return node.predicted_class

        Y = []
        for xi in X :
            Y.append(predict_xi(xi))          
        return Y    



class  Node():
    """
    class that represents a node in the tree 
    """
    def __init__(self,parent_node):
        self.right=None 
        self.left=None
        self.parent = parent_node
        self.leaf=False # boolean to tell if a node is a leaf (end node in the tree)

