import numpy as np
import scipy
import scipy.io
import scipy.linalg
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime 

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', unicode=True)
matplotlib.rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
matplotlib.rc('text.latex', preamble=r'\usepackage[english]{babel}')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rc('text.latex', preamble=r'\usepackage{bm}')

def HankelMatrix(X, L):
    N = X.shape[0]
    return scipy.linalg.hankel(X[ : N - L + 1], X[N - L : N])


def Svd(H, verbose=False):
    U, s, VT = np.linalg.svd(H)
    if verbose:
        print(s[:5])
    return U, s, VT


def FindKnnIdx(H, t, n_neighbors, verbose=False):
    norm_list = np.array([np.linalg.norm(row - H[t]) for row in H])
    knn_idx = norm_list.argsort()[:n_neighbors + 1]
    
    return knn_idx


def Projection(U, s, V):
    print(U.shape, s.shape, V.shape)
    L = V.shape[1]
    S = np.zeros((U.shape[0], L))
    S[:L, :L] = np.diag(s)
    
    return U.dot(S)


def PlotTsWithKnn(X, knn_idx, t_max=None, marker_size=70, savefig=False, filename=None):
    plt.figure(figsize=(16, 6))

    plot_idx = []
    for idx in knn_idx:
        if t_max is not None and idx < t_max or t_max is None:
            plot_idx.append(idx)
            
    # print(X.shape)
    # print(plot_idx)
    if t_max is None:
        right = X.shape[0]
    elif t_max > X.shape[0]:
        right = X.shape[0]
    else:
        right = t_max
        
    
    plt.plot(X[0 : right], zorder=0)
    plt.scatter(plot_idx, X[plot_idx], color='red', zorder=1, s=marker_size)
    plt.scatter(plot_idx[0], X[plot_idx[0]], color='black', zorder=1, s=marker_size)

    plt.xticks(size=16)
    plt.yticks(size=16)
    
    if savefig and filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()


def PlotKnnOnPhaseTrajectories(Pr_list, savefig=False, filename=None, xticks=None, xticks_size=22, i_list=[0]):
    if xticks is not None:
        assert len(xticks) == len(Pr_list), "len(xy_labels) != len(Pr_list)"
    
    n = len(Pr_list)
    fig, ax = plt.subplots(1, n, figsize=(8 * n, 6))

    for i, Pr in enumerate(Pr_list):
        coords, knn_idx, title, idx, lims = Pr
        knn_pr = coords[knn_idx]
        ax[i].plot(coords[:, idx[0]][lims[0]:lims[1]], coords[:, idx[1]][lims[0]:lims[1]], zorder=0, lw=1)

        ax[i].scatter(knn_pr[:, idx[0]], knn_pr[:, idx[1]], 
                      color='red', zorder=5, facecolors='none', s=80, lw=2)
        ax[i].scatter(knn_pr[0, idx[0]], knn_pr[0, idx[1]], 
                      color='black', zorder=6, facecolors='none', s=80, lw=2)
        
        if xticks is None:
            xticks = [('$y_1$', '$y_2$') for i in range(n)]
        ax[i].set_xlabel(xticks[i][0], fontsize=xticks_size)
        ax[i].set_ylabel(xticks[i][1], fontsize=xticks_size)
        ax[i].set_title(str(len(knn_idx)) + ' neighbors, ' + title, fontsize=20)
        
    # New code
    
    if False:
        def connect_points(i1, i2):
            fig = plt.gcf()
            
            ax0tr = ax[i1].transData # Axis 0 -> Display
            ax1tr = ax[i2].transData # Axis 1 -> Display
            figtr = fig.transFigure.inverted() # Display -> Figure
            
            axes = [ax0tr, ax1tr]
            
            def get_pts(index):
                coords, knn_idx, title, idx, lims = Pr_list[index]
                knn_pr = coords[knn_idx]
                pts = zip(knn_pr[:, idx[0]], knn_pr[:, idx[1]])
                transformed = []
                for pt in pts:
                    pt_transformed = figtr.transform(axes[index].transform((pt[0], pt[1])))
                    transformed.append(pt_transformed)
                return transformed
 
            pts_1 = get_pts(0)
            pts_2 = get_pts(1)
            
            for i in i_list:
                arrow = matplotlib.patches.FancyArrowPatch(
                    pts_1[i], pts_2[i], transform=fig.transFigure,  # Place arrow in figure coord system
                    fc = "yellow", ec=None, arrowstyle='simple',
                    mutation_scale = 20.0, zorder=1)
                fig.patches.append(arrow)
            
        connect_points(0, 1)

    
    # ...

    plt.setp(ax[0].get_xticklabels(), fontsize=xticks_size)
    plt.setp(ax[0].get_yticklabels(), fontsize=xticks_size)
    plt.setp(ax[1].get_xticklabels(), fontsize=xticks_size)
    plt.setp(ax[1].get_yticklabels(), fontsize=xticks_size)

    if savefig and filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show() 
