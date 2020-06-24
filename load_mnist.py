# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio

def load_mnist_idx(data_type):
       data_dir = 'datasets/mnist/'
       fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
       fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       trY = loaded[8:].reshape((60000)).astype(np.float)
       fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
       fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
       loaded = np.fromfile(file=fd,dtype=np.uint8)
       teY = loaded[8:].reshape((10000)).astype(np.float)
       trY = np.asarray(trY)
       teY = np.asarray(teY)
       if data_type == "train":
           X = trX[0:50000,:,:,:]
           y = trY[0:50000].astype(np.int)
       elif data_type == "test":
           X = teX
           y = teY.astype(np.int)
       elif data_type == "val":
           X = trX[50000:60000,:,:,:]
           y = trY[50000:60000].astype(np.int)
       idxUse = np.arange(0,y.shape[0])
       seed = 547
       np.random.seed(seed)
       np.random.shuffle(X)
       np.random.seed(seed)
       np.random.shuffle(y)
       np.random.seed(seed)
       np.random.shuffle(idxUse)

       return X/255.,y,idxUse
   
def load_mnist_classSelect(data_type,class_use,newClass):
    
    X, Y, idx = load_mnist_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx

def load_fashion_mnist_idx(data_type):
    import mnist_reader
    data_dir = 'datasets/fmnist/'
    if data_type == "train":
        X, y = mnist_reader.load_mnist(data_dir, kind='train')
    elif data_type == "test" or data_type == "val":
        X, y = mnist_reader.load_mnist(data_dir, kind='t10k')
        if data_type == "test":
            X = X[:4000,:]
            y = y[:4000]
        else:
            X = X[4000:,:]
            y = y[4000:]
    X = X.reshape((X.shape[0],28,28,1))        
    X = X.astype(np.float)
    y = y.astype(np.int)
    
    idxUse = np.arange(0,y.shape[0])
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    np.random.seed(seed)
    np.random.shuffle(idxUse)
    return X/255.,y,idxUse

def load_fashion_mnist_classSelect(data_type,class_use,newClass):
    
    X, Y, idx = load_fashion_mnist_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx
    

def load_svhn_idx(data_type):
       data_dir = 'datasets/SVHN/'
       if data_type == "train":
           data = sio.loadmat(data_dir + 'train_32x32.mat')
           X = data['X'].astype(np.float)
           y = data['y'].astype(np.int)
       elif data_type == "val": 
           data = sio.loadmat(data_dir + 'test_32x32.mat')
           X = data['X']
           X = X[:,:,:,:10000].astype(np.float)
           y = data['y']
           y = y[:10000].astype(np.int)
       elif data_type == "test": 
           data = sio.loadmat(data_dir + 'test_32x32.mat')
           X = data['X']
           X = X[:,:,:,10000:].astype(np.float)
           y = data['y']
           y = y[10000:].astype(np.int)
       
       X = X.transpose(3,2,0,1)
       zero_idx = np.where(y == 10)[0]
       y[zero_idx] = 0

       idxUse = np.arange(0,y.shape[0])
       seed = 547
       np.random.seed(seed)
       np.random.shuffle(X)
       np.random.seed(seed)
       np.random.shuffle(y)
       np.random.seed(seed)
       np.random.shuffle(idxUse)

       return X/255.,y,idxUse
   
def load_svhn_classSelect(data_type,class_use,newClass):
    
    X, Y, idx = load_svhn_idx(data_type)
    class_idx_total = np.zeros((0,0))
    Y_use = Y
    
    count_y = 0
    for k in class_use:
        class_idx = np.where(Y[:]==k)[0]
        Y_use[class_idx] = newClass[count_y]
        class_idx_total = np.append(class_idx_total,class_idx)
        count_y = count_y +1
        
    class_idx_total = np.sort(class_idx_total).astype(int)

    X = X[class_idx_total,:,:,:]
    Y = Y_use[class_idx_total]
    return X,Y,idx
