import numpy as np



def CM(real,predicted):
    matrix = np.zeros((2,2),dtype='int32')
    p = np.array(predicted).flatten()
    r = np.array(real).flatten()
    matrix[1,1] = len(r[(p==r) & (p==1)])        #true positive
    matrix[0,0] = len(r[(p==r) & (p==0)])       #true negative
    matrix[0,1] = len(r[(p!=r) & (r==1)])       # false negative
    matrix[1,0] = (p.shape[0])-(matrix[0,0]+matrix[1,1]+matrix[1,0]) # false positive

    return matrix 