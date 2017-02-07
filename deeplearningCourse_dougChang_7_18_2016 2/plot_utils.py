import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
import pylab as pl

from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['red','yellow','blue' ])

def make_plots(f, size = (10,7),history = None, lim=8, inc=0.25, levels=10, jitter=True, cmap=cm.RdBu):
    def build_mesh(lim,inc,f):
        X = np.arange(-lim, lim, inc)
        Y = np.arange(-lim, lim, inc)
        X, Y = np.meshgrid(X, Y)
        rowRange = range(X.shape[0])
        colRange = range(X.shape[1])
        Z= np.zeros((X.shape[0],X.shape[1]))
        for i in rowRange:
            for j in colRange:
                Z[i,j] = f(np.array([X[i,j],Y[i,j]]))    
        return X,Y,Z
    
    X,Y,Z = build_mesh(lim,inc,f)  
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    
    if history is not None:
        xlist = np.array(map(lambda val: val[0][0],history))
        ylist = np.array(map(lambda val: val[0][1],history))
        zlist = np.array(map(lambda val: val[1],history))
        
        if jitter:
            noise = 0.2
            xlist = xlist + np.random.normal(0, noise, size=len(xlist))
            ylist = ylist + np.random.normal(0, noise, size=len(ylist))
            zlist = zlist + np.random.normal(0, noise, size=len(zlist))
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
            linewidth=0, antialiased=False,cmap=cmap,alpha=0.3)
    ax.contour(X,Y,Z,zdir='z',offset=-2)
    if history is not None:
        ax.plot(xlist,ylist,zlist, color='r', marker='o', label='Gradient decent')
    plt.show()
    
    fig = plt.figure(figsize=size)        
    plt.contour(X, Y, Z,levels)  
   
    cs = pl.contour(X,Y,Z,levels,zdir='z',offset=-2)
    pl.clabel(cs,inline=1,fontsize=10)
    if history is not None:    
        pl.plot(xlist, ylist, color='r', marker='o',alpha=0.8)
        pl.plot(xlist[0], ylist[0],'go', alpha=0.5, markersize=10)
        pl.text(xlist[0], ylist[0],'  start', va='center')
        pl.plot(xlist[-1], ylist[-1],'ro', alpha=0.5, markersize=10)
        pl.text(xlist[-1], ylist[-1],'  stop', va='center')
    plt.show()
    



def plot_classify(X,y,predict,cm= pl.cm.Spectral,use_prob=False,scatter_cm=pl.cm.Spectral):
    step = .02  
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))
    m, n = np.c_[xx.ravel(), yy.ravel()].shape
    mesh = np.c_[xx.ravel(), yy.ravel()]
    #mesh = add_bias(mesh)
    if (use_prob):
        Z = predict(mesh)[1][:,1]
    else:
        Z = predict(mesh)[0]
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    markerSize = 40

    ax.contourf(xx, yy, Z, cmap=cm, alpha=.9)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=scatter_cm,s = markerSize)
    plt.show()
    


