# -*- coding: utf-8 -*-
import sklearn.decomposition
import sklearn.linear_model
from load_mnist import load_mnist_idx
import numpy as np
import matplotlib.pyplot as plt

def lr_mnist(y0=1, y1=7, pca_dim=3, debug=False):
    # implements PCA, then logistic regression, on two MNIST classes
    # class defined directed as the classified digit
    
    # load data
    trX_all, trY_all, tridx_all = load_mnist_idx('train',10)
    vaX_all, vaY_all, vaidx_all = load_mnist_idx('val',10)
    teX_all, teY_all, teidx_all = load_mnist_idx('test',10)
    
    # vectorize
    trXv_all = trX_all.reshape(trX_all.shape[0],trX_all.shape[1]*trX_all.shape[2])
    vaXv_all = vaX_all.reshape(vaX_all.shape[0],vaX_all.shape[1]*vaX_all.shape[2])
    teXv_all = teX_all.reshape(teX_all.shape[0],teX_all.shape[1]*teX_all.shape[2])
    
    # mask
    trXv = trXv_all[np.bitwise_or(trY_all == y0, trY_all == y1),:]
    trY = trY_all[np.bitwise_or(trY_all == y0, trY_all == y1)]
    tridx = tridx_all[np.bitwise_or(trY_all == y0, trY_all == y1)]
    
    vaXv = vaXv_all[np.bitwise_or(vaY_all == y0, vaY_all == y1),:]
    vaY = vaY_all[np.bitwise_or(vaY_all == y0, vaY_all == y1)]
    vaidx = vaidx_all[np.bitwise_or(vaY_all == y0, vaY_all == y1)]
    
    teXv = teX_all[np.bitwise_or(teY_all == y0, teY_all == y1),:]
    teY = teY_all[np.bitwise_or(teY_all == y0, teY_all == y1)]
    teidx = teidx_all[np.bitwise_or(teY_all == y0, teY_all == y1)]
    
    # standardize data
    trMean = np.mean(trXv, axis=0)
    trStd = np.std(trXv, axis=0)
    normalizer = trStd + (trStd==0).astype(int)
    trXvs = (trXv - trMean) / normalizer # broadcasting
    
    # PCA
    pca = sklearn.decomposition.PCA(n_components=pca_dim)
    pca.fit(trXvs)
    trXv_princ = pca.transform(trXvs)
    
    P = pca.transform(np.eye(trXvs.shape[1])) # get transform matrix
        
    if debug:
        # visualize first two principal components
        plt.figure(0)
        plt.scatter(trXv_princ[trY == y0,0],trXv_princ[trY == y0,1],c='r')
        plt.scatter(trXv_princ[trY == y1,0],trXv_princ[trY == y1,1],c='b')
        
        for c in range(min(pca_dim,10)):
            plt.figure(c+1)
            plt.imshow(P[:,c].reshape(trX_all.shape[1],trX_all.shape[2]))
    
    # Logistic Regression
    lr = sklearn.linear_model.LogisticRegression(solver='liblinear').fit(trXv_princ,trY)
    tracc = lr.score(trXv_princ,trY)
    if debug: print 'Training accuracy: %f\n' % tracc
    
    ### convert classifier to hyperplane
    normMat = np.diag(1 / normalizer)
    normMeanVec = -np.matmul(normMat, trMean)
    
    # if x is raw data column vector, x_standardized = normMat * x + normMeanVec
    #                          x_PCA = P^T * x_standardized
    
    lr_weight = lr.coef_.T
    lr_bias = lr.intercept_
    
    # classifier result: lr_weight^T * x_PCA + lr_bias 
    #                  = lr_weight^T * P^T * x_standardized + lr_bias 
    #                  = lr_weight^T * P^T * (normMat * x + normMeanVec) + lr_bias 
    #                  = lr_weight^T * P^T * normMat * x + (lr_weight^T * P^T * normMeanVec + lr_bias)
    #                  = (normMat * P * lr_weight)^T * x + (lr_weight^T * P^T * normMeanVec + lr_bias)
    #                  = total_weight^T * x - total_bias
    
    total_weight = np.matmul(np.matmul(normMat,P), lr_weight).squeeze()
    total_bias = -(np.matmul(lr_weight.T,np.dot(P.T,normMeanVec)) + lr_bias)
    
    # process validation
    vaXvs = (vaXv - trMean) / normalizer 
    vaXv_princ = pca.transform(vaXvs)
    
    vaacc = lr.score(vaXv_princ,vaY)
    if debug: print 'Validation accuracy: %f\n' % vaacc
    
    # verify manual classifier
    y_pred_raw = np.sign(np.matmul(vaXv,total_weight) - total_bias)
    y_pred_man = np.zeros(y_pred_raw.shape)
    y_pred_man[y_pred_raw < 0] = y0
    y_pred_man[y_pred_raw > 0] = y1
    
    vaacc_man = np.mean(y_pred_man == vaY)
    if debug: print 'Manual validation accuracy: %f\n' % vaacc_man
    
    # process testing
    # teXvs = (teXv - teMean) / normalizer 
    # teXv_princ = pca.transform(teXvs)
    
    return total_weight,total_bias,vaacc
