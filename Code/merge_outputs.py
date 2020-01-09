#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:33:23 2019

@author: vr79958h
"""

import os
import numpy as np

fold = '../Outputs'
nonnormdiagnoncons1 = dict()
nonnormnondiagnoncons1 = dict()
normdiagnoncons1 = dict()
normnondiagnoncons1 = dict()
nonnormdiagcons1 = dict()
nonnormnondiagcons1 = dict()
normdiagcons1 = dict()
normnondiagcons1 = dict()
nonnormdiagnoncons3 = dict()
nonnormnondiagnoncons3 = dict()
normdiagnoncons3 = dict()
normnondiagnoncons3 = dict()
nonnormdiagcons3 = dict()
nonnormnondiagcons3 = dict()
normdiagcons3 = dict()
normnondiagcons3 = dict()


def var_name(var):
    for name, value in globals().items():
        if value is var:
            return name
    return '?????'


def do(file, list_, dic, index):
    if list_[index] not in dic:
        dic[list_[index]] = {'f1': [], 'TN': [], 'FP': [], 'FN': [], 'TP': [],
                             'mu': [], 'c': []}
    if list_[-2] == 'c':
        with open(f'./{fold}/{file}', 'r') as f:
            content = f.read()
            content = content.split('\n')
            dic[list_[index]]['f1'].append(float(content[1][2:]))
            dic[list_[index]]['TN'].append(float(content[2][2:]))
            dic[list_[index]]['FP'].append(float(content[3][2:]))
            dic[list_[index]]['FN'].append(float(content[4][2:]))
            dic[list_[index]]['TP'].append(float(content[5][2:]))
            dic[list_[index]]['mu'].append(float(content[6][2:]))
            dic[list_[index]]['c'].append(float(content[7][2:]))
    return dic


# recuperation
for file in os.listdir(f'./{fold}'):
    list_ = file.split('_')
    if list_[-1] in ['prediction.csv', 'loss.csv', 'measure.csv', 'L.csv']:
        pass
    elif file[:23] == '1NN_nonnorm_diag_cons1_':
        nonnormdiagcons1 = do(file, list_, nonnormdiagcons1, 4)
    elif file[:26] == '1NN_nonnorm_diag_noncons1_':
        nonnormdiagnoncons1 = do(file, list_, nonnormdiagnoncons1, 4)
    elif file[:26] == '1NN_nonnorm_nondiag_cons1_':
        nonnormnondiagcons1 = do(file, list_, nonnormnondiagcons1, 4)
    elif file[:29] == '1NN_nonnorm_nondiag_noncons1_':
        nonnormnondiagnoncons1 = do(file, list_, nonnormnondiagnoncons1, 4)
    elif file[:20] == '1NN_norm_diag_cons1_':
        normdiagcons1 = do(file, list_, normdiagcons1, 4)
    elif file[:23] == '1NN_norm_diag_noncons1_':
        normdiagnoncons1 = do(file, list_, normdiagnoncons1, 4)
    elif file[:23] == '1NN_norm_nondiag_cons1_':
        normnondiagcons1 = do(file, list_, normnondiagcons1, 4)
    elif file[:26] == '1NN_norm_nondiag_noncons1_':
        normnondiagnoncons1 = do(file, list_, normnondiagnoncons1, 4)
    elif file[:23] == '3NN_nonnorm_diag_cons1_':
        nonnormdiagcons3 = do(file, list_, nonnormdiagcons3, 4)
    elif file[:26] == '3NN_nonnorm_diag_noncons1_':
        nonnormdiagnoncons3 = do(file, list_, nonnormdiagnoncons3, 4)
    elif file[:26] == '3NN_nonnorm_nondiag_cons1_':
        nonnormnondiagcons3 = do(file, list_, nonnormnondiagcons3, 4)
    elif file[:29] == '3NN_nonnorm_nondiag_noncons1_':
        nonnormnondiagnoncons3 = do(file, list_, nonnormnondiagnoncons3, 4)
    elif file[:20] == '3NN_norm_diag_cons1_':
        normdiagcons3 = do(file, list_, normdiagcons3, 4)
    elif file[:23] == '3NN_norm_diag_noncons1_':
        normdiagnoncons3 = do(file, list_, normdiagnoncons3, 4)
    elif file[:23] == '3NN_norm_nondiag_cons1_':
        normnondiagcons3 = do(file, list_, normnondiagcons3, 4)
    elif file[:26] == '3NN_norm_nondiag_noncons1_':
        normnondiagnoncons3 = do(file, list_, normnondiagnoncons3, 4)

# mean
for dic in [nonnormdiagnoncons1, nonnormnondiagnoncons1, normdiagnoncons1,
            normnondiagnoncons1, nonnormdiagcons1, nonnormnondiagcons1,
            normdiagcons1, normnondiagcons1,
            nonnormdiagnoncons3, nonnormnondiagnoncons3, normdiagnoncons3,
            normnondiagnoncons3, nonnormdiagcons3, nonnormnondiagcons3,
            normdiagcons3, normnondiagcons3]:
    for k in dic.keys():
        for key in ['f1', 'TN', 'FP', 'FN', 'TP', 'mu', 'c']:
            if key == 'f1' and len(dic[k][key]) != 5:
                print(k, var_name(dic), len(dic[k][key]))
            dic[k][key] = np.mean(np.array([dic[k][key]]))

# folder creation
os.makedirs(f'../CSV/', exist_ok=True)

# write
for di in ['nonnormdiagnoncons1', 'nonnormnondiagnoncons1', 'normdiagnoncons1',
           'normnondiagnoncons1', 'nonnormdiagcons1', 'nonnormnondiagcons1',
           'normdiagcons1', 'normnondiagcons1',
           'nonnormdiagnoncons3', 'nonnormnondiagnoncons3', 'normdiagnoncons3',
           'normnondiagnoncons3', 'nonnormdiagcons3', 'nonnormnondiagcons3',
           'normdiagcons3', 'normnondiagcons3']:
    dic = eval(di)
    if len(dic) != 0:
        output = ""
        for k in dic.keys():
            for key in ['f1', 'TN', 'FP', 'FN', 'TP', 'mu', 'c']:
                output += f'{k},{key},{dic[k][key].round(3)}\n'
        with open(f'../CSV/{di}.csv', 'w') as f:
            f.write(output)
