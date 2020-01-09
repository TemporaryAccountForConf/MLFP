#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os

from functions import data_recovery

from mlfp import mlfp

from skopt.space import Real
from skopt import gp_minimize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import numpy as np

import pandas as pd

from multiprocessing import Pool

import warnings
from sklearn.exceptions import DataConversionWarning

# UserWarning: The objective has been evaluated at this point before.
warnings.filterwarnings("ignore", category=UserWarning)

# DataConversionWarning: Data with input dtype int64 was converted to float64
# by StandardScaler.
warnings.filterwarnings("ignore", category=DataConversionWarning)

beta = 1  # If we want change the f-measure


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="autompg", help=('dataset \
                    in"satimage","pageblocks","abalone20","abalone17",\
                    "abalone8","segmentation","wine4","yeast6","yeast3",\
                    "german","vehicle","pima","balance","autompg","libras",\
                    "iono","glass","wine","hayes"'))

parser.add_argument('--seed', type=int, default=123, help='seed for the \
                    randomness, in [12,123,1234,12345,123456]')

parser.add_argument('--nb_nn', type=int, default=1, help='number of neighbors \
                    for kNN')

parser.add_argument('--mu', nargs=2, type=float, default=[0.0, 1.0], help='min\
                    and max values for mu on MLFP')

parser.add_argument('--c', nargs=2, type=float, default=[0.0, 1.0], help='min \
                    and max values for c on MLFP')

parser.add_argument('--nb_cv', type=int, default=10, help='number of fold for \
                    the cross validation')

parser.add_argument('--normalization', type=str2bool, default=True, help='\
                    perform a normalization')

parser.add_argument('--pca', type=str2bool, default=True, help='perform a PCA\
                    for dimensonality reduction')

parser.add_argument('--diag', type=str2bool, default=True, help='constrain L \
                    to be diagonal')

parser.add_argument('--cons1', type=str2bool, default=True, help='constrain \
                    eigenvalues of L to be less than one')

opt = parser.parse_args()

np.random.seed(seed=opt.seed)

print(opt)

date = time.strftime("%Y_%b_%d_%H_%M_%S", time.localtime(round(time.time())))

print(date)

##########################
#                        #
# File Names Preparation #
#                        #
##########################

opt.name = str(opt.nb_nn) + 'NN'

if opt.normalization:
    opt.name = opt.name + '_norm'
else:
    opt.name = opt.name + '_nonnorm'

if opt.pca:
    opt.name = opt.name + '_PCA'

if opt.diag:
    opt.name = opt.name + '_diag'
else:
    opt.name = opt.name + '_nondiag'

if opt.cons1:
    opt.name = opt.name + '_cons1'
else:
    opt.name = opt.name + '_noncons1'

opt.name = opt.name + f'_{opt.dataset}_{opt.seed}'

###################
#                 #
# Folder Creation #
#                 #
###################

os.makedirs(f'../Plots/', exist_ok=True)

os.makedirs(f'../Outputs/', exist_ok=True)

#############
#           #
# Functions #
#           #
#############


def learning(param, ctrl=False):

    params, X_tr, y_tr, X_va, y_va = param
    X_te = X[test_index]
    y_te = y[test_index]

    #################
    #               #
    # Normalization #
    #               #
    #################

    if opt.normalization:
        normalizer = StandardScaler()
        normalizer.fit(X_tr)
        X_tr = normalizer.transform(X_tr)
        X_va = normalizer.transform(X_va)
        X_te = normalizer.transform(X_te)

    #######
    #     #
    # PCA #
    #     #
    #######

    if opt.pca:
        pca = PCA(n_components=dim)
        pca.fit(X_tr)
        nb_pca = np.where(
                (np.cumsum(pca.explained_variance_ratio_) > 0.999) == 1
                )[0][0]+1
        nb_pca = 2
        pca = PCA(n_components=nb_pca)
        pca.fit(X_tr)
        X_tr = pca.transform(X_tr)
        X_va = pca.transform(X_va)
        X_te = pca.transform(X_te)

    gml = mlfp(k=opt.nb_nn, randomState=opt.seed, mu=params[0],
               c=params[1], diag=opt.diag, const1=opt.cons1)

    # For controling overfitting
    gml.ctrl = ctrl

    if gml.ctrl:
        gml.X_train = X_tr
        gml.X_valid = X_va
        gml.y_train = y_tr
        gml.y_valid = y_va
        gml.X_test = X_te
        gml.y_test = y_te
        gml.store_f_m = []
        gml.store_loss = []

    gml.fit(X_tr, y_tr)

    if gml.ctrl:
        df = pd.DataFrame(gml.store_loss)
        df.columns = ['Train', 'Valid', 'Test']
        df.to_csv(f'../Outputs/{opt.name}_loss.csv')
        df = pd.DataFrame(gml.store_L)
        df.to_csv(f'../Outputs/{opt.name}_all_L.csv')
    if gml.ctrl:
        gml.ctrl = False

    pred = gml.predict(X_va)

    TN, FP, FN, TP = confusion_matrix(y_va, pred).ravel()

    f_measure = (1+beta**2)*TP / ((1+beta**2)*TP+(beta**2)*FN+FP)

    if ctrl:
        df = pd.DataFrame(gml.store_f_m)
        df.columns = ['Train', 'Valid', 'Test']
        df.to_csv(f'../Outputs/{opt.name}_ctrl_f_measure.csv')

        df = pd.DataFrame(pred)
        df.to_csv(f'../Outputs/{opt.name}_prediction.csv')

        if opt.pca:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap

            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
            cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

            h = 0.05
            x_min, x_max = X_tr[:, 0].min()-1, X_tr[:, 0].max()+1
            y_min, y_max = X_tr[:, 1].min()-1, X_tr[:, 1].max()+1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            z = gml.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            plt.figure()
            plt.pcolormesh(xx, yy, z, cmap=cmap_light)
            plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=cmap_bold, s=.1)
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')

            plt.savefig(f'../Plots/{opt.name}_classifier.pdf')
            plt.close()

        return [round(f_measure, 3), TN, FP, FN, TP, gml]

    return f_measure


def objective(params):

    with Pool() as pool:
        f_measure = pool.map_async(
                learning,
                [(params,
                  X[train_index][trainv_index],
                  y[train_index][trainv_index],
                  X[train_index][valid_index],
                  y[train_index][valid_index],
                  ) for trainv_index, valid_index in kf.split(
                          X[train_index],
                          y[train_index])]).get()

    f_measure = np.mean(f_measure)
    print(params, f_measure)
    return 1 - f_measure


#################
#               #
# Data Recovery #
#               #
#################

X, y, dim = data_recovery(opt, date)

####################
#                  #
# Train Test Split #
#                  #
####################

kf = StratifiedKFold(n_splits=5, random_state=opt.seed, shuffle=True)
kf = kf.split(X, y)
train_index, test_index = kf.__next__()

########
#      #
# MLFP #
#      #
########

######
#    #
# CV #
#    #
######

kf = StratifiedKFold(n_splits=opt.nb_cv, random_state=opt.seed, shuffle=True)

space = [Real(opt.mu[0], opt.mu[1], name='mu'),
         Real(opt.c[0], opt.c[1], name='c')]

res_gp = gp_minimize(objective, space, n_calls=400, random_state=opt.seed,
                     n_jobs=-1)

mu = res_gp.x[0]
c = res_gp.x[1]

############
#          #
# Learning #
#          #
############

f_measure, TN, FP, FN, TP, gml = learning(((mu, c),
                                           X[train_index], y[train_index],
                                           X[test_index], y[test_index]),
                                          ctrl=True)
print(f'MLFP -> {f_measure} ( mu = {mu} ), ( c = {c} )')
pd.DataFrame([f_measure, TN, FP, FN, TP, mu, c]).to_csv(
        f'../Outputs/{opt.name}_f_measure_TN_FP_FN_TP_mu_c.csv')

date = time.strftime("%Y_%b_%d_%H_%M_%S", time.localtime(round(time.time())))

print(date)
